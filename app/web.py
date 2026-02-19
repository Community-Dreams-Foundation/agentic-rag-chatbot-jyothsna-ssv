import os
# Use a separate Chroma collection for the web app so "make sanity" doesn't preload our index
os.environ["CHROMA_COLLECTION_NAME"] = "web_documents"

import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from app.rag import index_document, answer_with_citations
from app.memory import analyze_memory_signal, persist_memory

project_root = Path(__file__).parent.parent
app = Flask(__name__, 
            template_folder=str(project_root / 'templates'),
            static_folder=str(project_root / 'static'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.after_request
def after_request(response):
    """Add CORS headers so the front end can call the API from the same host."""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'html', 'htm', 'md'}


def allowed_file(filename):
    """Only allow extensions we know how to parse (txt, pdf, html, md)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main chat page."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}<br>Template folder: {app.template_folder}", 500


@app.route('/health')
def health():
    """Simple health check so we know the server and template are OK."""
    return jsonify({
        'status': 'ok',
        'template_folder': app.template_folder,
        'template_exists': Path(app.template_folder, 'index.html').exists()
    })


@app.route('/test')
def test():
    return """
    <html>
    <head><title>Web App Test</title></head>
    <body style="font-family: Arial; padding: 40px; background: #f0f0f0;">
        <h1>✅ Web App is Working!</h1>
        <p>If you see this page, the Flask server is running correctly.</p>
        <p><a href="/">Go to main app</a></p>
        <hr>
        <h2>Debug Info:</h2>
        <pre>
Template folder: {template_folder}
Template exists: {template_exists}
        </pre>
    </body>
    </html>
    """.format(
        template_folder=app.template_folder,
        template_exists=Path(app.template_folder, 'index.html').exists()
    )


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Save uploaded file(s), run them through the RAG pipeline, and return indexing stats."""
    if 'file' not in request.files:
        return jsonify({'error': 'Please select a file to upload'}), 400
    
    files = request.files.getlist('file')
    source_tag = request.form.get('source_tag', '').strip()
    reindex_all = request.form.get('reindex_all', 'false').lower() == 'true'
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'Please select at least one file to upload'}), 400
    
    results = []
    total_files_parsed = 0
    total_chunks_created = 0
    
    for file in files:
        if file.filename == '':
            continue
            
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'success': False,
                'error': f'I can only process files in these formats: {", ".join(ALLOWED_EXTENSIONS)}. Please upload a file with one of these extensions.'
            })
            continue
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            tag = source_tag if source_tag else None
            
            try:
                stats = index_document(filepath, source_tag=tag)
                total_files_parsed += stats['files_parsed']
                total_chunks_created += stats['chunks_created']
                results.append({
                    'filename': filename,
                    'success': True,
                    'stats': stats
                })
            except ImportError as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e),
                    'hint': 'Install missing dependency: pip install pypdf (for PDF) or beautifulsoup4 (for HTML)'
                })
            except ValueError as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
            except FileNotFoundError as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': f'I had trouble processing this file: {str(e)}'
            })
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    return jsonify({
        'success': len(successful) > 0,
        'files_parsed': total_files_parsed,
        'chunks_created': total_chunks_created,
        'indexed': True,
        'results': results,
        'summary': {
            'total_files': len(files),
            'successful': len(successful),
            'failed': len(failed)
        }
    })


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Take a question, run RAG, return answer plus citations and retrieved chunks."""
    data = request.get_json()
    question = data.get('question', '').strip()
    top_k = int(data.get('top_k', 5))
    retrieval_mode = data.get('retrieval_mode', 'hybrid')
    
    if not question:
        return jsonify({'error': 'Please ask a question'}), 400
    
    try:
        use_hybrid = (retrieval_mode == 'hybrid')
        result = answer_with_citations(question, top_k=top_k, use_hybrid=use_hybrid)
        
        from app.rag import retrieve_chunks
        documents, metadatas = retrieve_chunks(
            question, 
            top_k=top_k, 
            use_hybrid=use_hybrid, 
            rerank=True
        )
        
        retrieved_chunks = []
        for doc, meta in zip(documents[:top_k], metadatas[:top_k]):
            chunk_id = meta.get('locator', 'unknown')
            retrieved_chunks.append({
                'chunk_id': chunk_id,
                'source': meta.get('source', 'unknown'),
                'locator': str(chunk_id),
                'text': doc[:500]
            })
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'citations': result['citations'],
            'retrieved_chunks': retrieved_chunks
        })
    except Exception as e:
        return jsonify({'error': f'I encountered an error while processing your question: {str(e)}'}), 500


@app.route('/api/memory', methods=['POST'])
def add_memory():
    """Parse user text for memory signals and append to USER_MEMORY.md / COMPANY_MEMORY.md."""
    data = request.get_json(silent=True) or {}
    user_input = (data.get('input') or '').strip()
    
    if not user_input:
        return jsonify({'error': 'Please tell me what you\'d like me to remember'}), 400
    
    try:
        decisions = analyze_memory_signal(user_input)
        if not decisions and user_input:
            from app.memory import _looks_sensitive
            if not _looks_sensitive(user_input):
                decisions = [{
                    "should_write": True,
                    "target": "USER",
                    "summary": f"User note: {user_input}",
                    "confidence": 0.8
                }]
        memory_writes = persist_memory(decisions)
        return jsonify({
            'success': True,
            'memory_writes': memory_writes,
            'decisions': decisions,
            'message': f'Processed {len(memory_writes)} memory entries' if memory_writes else 'No memory entries written'
        })
    except Exception as e:
        return jsonify({'error': f'I encountered an error while processing your memory request: {str(e)}'}), 500


@app.route('/api/memory/view', methods=['GET'])
def view_memory():
    """Return the current contents of both memory files for the UI to show."""
    try:
        user_memory = Path('USER_MEMORY.md')
        company_memory = Path('COMPANY_MEMORY.md')
        
        user_content = user_memory.read_text(encoding='utf-8') if user_memory.exists() else '# Memory Log\n\n'
        company_content = company_memory.read_text(encoding='utf-8') if company_memory.exists() else '# Memory Log\n\n'
        
        return jsonify({
            'success': True,
            'user_memory': user_content,
            'company_memory': company_content
        })
    except Exception as e:
        return jsonify({'error': f'Error reading memory: {str(e)}'}), 500


@app.route('/api/files', methods=['GET'])
def list_files():
    """List filenames in the uploads folder so the UI knows what's indexed."""
    try:
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        files = [f.name for f in upload_dir.iterdir() if f.is_file() and allowed_file(f.name)]
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500


if __name__ == '__main__':
    import sys
    
    port = int(os.environ.get('PORT', 5001))
    
    print("=" * 60)
    print("🚀 Starting Agentic RAG Chatbot Web App")
    print("=" * 60)
    print(f"📁 Template folder: {app.template_folder}")
    print(f"✅ Template exists: {Path(app.template_folder, 'index.html').exists()}")
    print()
    print("🌐 Server starting...")
    print(f"   Open your browser to: http://localhost:{port}")
    print(f"   Test page: http://localhost:{port}/test")
    print(f"   Health check: http://localhost:{port}/health")
    print()
    print("   If you get 403 Forbidden:")
    print("   1. Make sure no other app is using port", port)
    print("   2. Try: PORT=5001 python -m app.web")
    print("   3. Check browser console (F12) for errors")
    print()
    print("   Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    try:
        app.run(debug=True, host='127.0.0.1', port=port, use_reloader=False)
    except OSError as e:
        if 'Address already in use' in str(e):
            print(f"\n❌ ERROR: Port {port} is already in use!")
            print(f"   Try: PORT=5001 python -m app.web")
            print(f"   Or kill the process: lsof -ti:{port} | xargs kill")
            sys.exit(1)
        else:
            raise
