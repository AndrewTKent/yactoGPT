# inference/server.py
from typing import Dict, Any
import json
from flask import Flask, request, jsonify, Response

from .generator import Generator


class InferenceServer:
    """REST API server for inference."""
    
    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any]
    ):
        """Initialize inference server.
        
        Args:
            model_path: Path to model checkpoint
            config: Server configuration
        """
        self.app = Flask(__name__)
        self.config = config
        
        # Load generator
        self.generator = Generator(
            model_path=model_path,
            device=config['inference'].get('device', 'auto'),
            dtype=config['inference'].get('dtype', 'float16')
        )
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
        
        @self.app.route('/generate', methods=['POST'])
        def generate():
            data = request.json
            
            # Get parameters
            prompt = data.get('prompt', '')
            max_new_tokens = data.get('max_new_tokens', self.config['inference']['max_new_tokens'])
            temperature = data.get('temperature', self.config['inference']['temperature'])
            top_k = data.get('top_k', self.config['inference']['top_k'])
            top_p = data.get('top_p', self.config['inference']['top_p'])
            num_samples = data.get('num_samples', 1)
            
            # Generate
            try:
                results = self.generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=num_samples
                )
                
                return jsonify({
                    'prompt': prompt,
                    'generated': results,
                    'parameters': {
                        'max_new_tokens': max_new_tokens,
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p
                    }
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/generate/stream', methods=['POST'])
        def generate_stream():
            data = request.json
            
            # Get parameters
            prompt = data.get('prompt', '')
            max_new_tokens = data.get('max_new_tokens', self.config['inference']['max_new_tokens'])
            temperature = data.get('temperature', self.config['inference']['temperature'])
            top_k = data.get('top_k', self.config['inference']['top_k'])
            top_p = data.get('top_p', self.config['inference']['top_p'])
            
            def stream():
                for token in self.generator.generate_stream(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    
            return Response(stream(), mimetype='text/event-stream')
            
    def run(self):
        """Run the server."""
        self.app.run(
            host=self.config['api']['host'],
            port=self.config['api']['port'],
            debug=False
        )
