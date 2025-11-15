import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import io
from flask_restx import Api, Resource, fields

# Add model directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

# Import necessary classes
from indextts.infer_v2 import IndexTTS2

path_checkpoints = "./checkpoints"
path_audioExamples = "./examples"
defaultAudioExampleFile = "voice_01.wav"
path_outputs = "./outputs"

class AudioGenerator:
    def __init__(self, model_dir=path_checkpoints):
        self.model_dir = model_dir
        # Initialize the model
        self.tts = IndexTTS2(
            model_dir=model_dir,
            cfg_path=os.path.join(model_dir, "config.yaml"),
            use_fp16=True,
            use_deepspeed=True,
            use_cuda_kernel=True
        )
    
    def generate_audio(self, text, audio_prompt=defaultAudioExampleFile):
        """Generate audio from text and return the file path"""
        # Generate filename with date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{path_outputs}/audio_{timestamp}.wav"
        
        try:
            result = self.tts.infer(
                spk_audio_prompt=f"{path_audioExamples}/{audio_prompt}",
                text=text,
                output_path=output_path,
                emo_audio_prompt=None,
                emo_alpha=0.65,
                emo_vector=None,
                use_emo_text=False,
                emo_text=None,
                use_random=False,
                verbose=False
            )
            return {"status": "success", "audio_path": result, "filename": f"audio_{timestamp}.wav"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate_audio_and_return_file(self, text, audio_prompt=defaultAudioExampleFile):
        """Generate audio and return the WAV file directly"""
        # Generate filename with date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{path_outputs}/audio_{timestamp}.wav"
        
        try:
            # Generate the audio
            result = self.tts.infer(
                spk_audio_prompt=f"{path_audioExamples}/{audio_prompt}",
                text=text,
                output_path=output_path,
                emo_audio_prompt=None,
                emo_alpha=0.65,
                emo_vector=None,
                use_emo_text=False,
                emo_text=None,
                use_random=False,
                verbose=False
            )
            
            # Check if file exists
            if not os.path.exists(output_path):
                return {"status": "error", "message": f"File not generated: {output_path}"}
            
            # Return file content as bytes
            with open(output_path, 'rb') as f:
                audio_data = f.read()
            
            return {
                "status": "success", 
                "audio_data": audio_data,
                "filename": f"audio_{timestamp}.wav",
                "content_type": "audio/wav"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Flask-RESTX configuration
app = Flask(__name__)
app.config['RESTX_VALIDATE'] = True

# Initialize API with Swagger
api = Api(app, 
    version='1.0',
    title='IndexTTS API',
    description='API for generating audio from text using IndexTTS',
    doc='/docs/'  # Path to access documentation
)

# Define models for Swagger
audio_model = api.model('AudioRequest', {
    'text': fields.String(required=True, description='Text to convert to audio'),
    'audio_prompt': fields.String(default=defaultAudioExampleFile, description='Name of reference audio file')
})

audio_response = api.model('AudioResponse', {
    'status': fields.String(description='Status of the operation'),
    'audio_path': fields.String(description='Path to generated file'),
    'filename': fields.String(description='Name of generated file')
})

audio_file_response = api.model('AudioFileResponse', {
    'status': fields.String(description='Status of the operation'),
    'filename': fields.String(description='Name of generated file'),
    'content_type': fields.String(description='Content type')
})

# Namespace to group routes
audio_ns = api.namespace('audio', description='Audio operations')

@audio_ns.route('/generate-audio')
class GenerateAudio(Resource):
    @api.doc('generate_audio')
    @api.expect(audio_model)
    @api.marshal_with(audio_response)
    def post(self):
        """Generate audio from text and return file path"""
        data = request.json
        text = data.get('text', '')
        audio_prompt = data.get('audio_prompt', defaultAudioExampleFile)
        
        result = generator.generate_audio(text, audio_prompt)
        return result

@audio_ns.route('/generate-audio-file')
class GenerateAudioFile(Resource):
    @api.doc('generate_audio_file')
    @api.expect(audio_model)
    def post(self):
        """Generate audio and return the WAV file directly"""
        data = request.json
        text = data.get('text', '')
        audio_prompt = data.get('audio_prompt', defaultAudioExampleFile)
        
        result = generator.generate_audio_and_return_file(text, audio_prompt)
        
        if result["status"] == "error":
            return jsonify(result), 500    
        
        # Return the file directly
        return send_file(
            io.BytesIO(result["audio_data"]),
            mimetype='audio/wav',
            as_attachment=True,
            download_name=result["filename"]
        )

@audio_ns.route('/generate-audio-file-stream')
class GenerateAudioFileStream(Resource):
    @api.doc('generate_audio_file_stream')
    @api.expect(audio_model)
    def post(self):
        """Alternative endpoint to return file with custom headers"""
        data = request.json
        text = data.get('text', '')
        audio_prompt = data.get('audio_prompt', defaultAudioExampleFile)
        
        result = generator.generate_audio_and_return_file(text, audio_prompt)
        
        if result["status"] == "error":
            return jsonify(result), 500    
        
        # Create response with custom headers
        response = jsonify({"status": "success", "filename": result["filename"]})
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = f'attachment; filename="{result["filename"]}"'
        
        return response

# Route for API health check
@api.route('/health')
class Health(Resource):
    @api.doc('health')
    def get(self):
        """Check API status"""
        return {'status': 'healthy'}

# Create audio generator
generator = AudioGenerator(path_checkpoints)

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)    
    app.run(debug=True, host='0.0.0.0', port=5000)
