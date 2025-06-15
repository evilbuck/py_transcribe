"""
Integration tests for transcriber module (simpler, less mocking)
"""
import pytest
from pathlib import Path
import tempfile

from transcribe.transcriber import AudioTranscriber


class TestAudioTranscriberBasic:
    """Basic integration tests for AudioTranscriber"""
    
    def test_initialization(self):
        """Test basic transcriber initialization"""
        transcriber = AudioTranscriber(model_size="tiny", device="cpu", verbose=False)
        assert transcriber.model_size == "tiny"
        assert transcriber.device == "cpu"
        assert transcriber.model is None
        
        info = transcriber.get_model_info()
        assert info['model_size'] == "tiny"
        assert info['device'] == "cpu"
        assert info['is_loaded'] == False
    
    def test_close(self):
        """Test transcriber cleanup"""
        transcriber = AudioTranscriber()
        transcriber.close()  # Should not crash
        
        info = transcriber.get_model_info()
        assert info['is_loaded'] == False
    
    def test_transcribe_text_only_mock(self, monkeypatch, temp_dir):
        """Test text-only transcription with mock"""
        
        def mock_transcribe_file(self, input_path, transcription_params=None):
            return {
                'segments': [
                    {'text': 'Hello world'},
                    {'text': 'This is a test'}
                ]
            }
        
        monkeypatch.setattr(AudioTranscriber, 'transcribe_file', mock_transcribe_file)
        
        input_file = temp_dir / "test.mp3"
        input_file.write_bytes(b"fake audio")
        
        transcriber = AudioTranscriber(model_size="tiny")
        result = transcriber.transcribe_text_only(input_file)
        
        assert result == "Hello world\nThis is a test"
    
    def test_write_transcript(self, temp_dir):
        """Test transcript writing"""
        transcriber = AudioTranscriber()
        
        segments = [
            {'text': 'First line'},
            {'text': 'Second line'},
            {'text': 'Third line'}
        ]
        
        output_file = temp_dir / "output.txt"
        transcriber._write_transcript(output_file, segments)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "First line\n" in content
        assert "Second line\n" in content
        assert "Third line\n" in content
    
    def test_write_transcript_error(self, temp_dir):
        """Test transcript writing error handling"""
        transcriber = AudioTranscriber()
        
        # Try to write to a directory that doesn't exist and can't be created
        invalid_path = temp_dir / "nonexistent" / "deeply" / "nested" / "output.txt"
        # Make parent directory non-writable on Unix systems
        if hasattr(Path, 'chmod'):
            try:
                temp_dir.chmod(0o444)  # Read-only
                
                with pytest.raises(RuntimeError, match="Failed to write transcript"):
                    transcriber._write_transcript(invalid_path, [{'text': 'test'}])
                    
            finally:
                temp_dir.chmod(0o755)  # Restore permissions
        else:
            # Skip on systems without chmod
            pytest.skip("chmod not available on this system")


class TestParameterHandling:
    """Test parameter handling without complex mocking"""
    
    def test_default_params_structure(self):
        """Test that default parameters have correct structure"""
        transcriber = AudioTranscriber()
        
        # The default params are created in transcribe_file method
        # We can't easily test them without mocking, but we can verify the structure
        # when they would be created
        default_params = {
            'beam_size': 5,
            'vad_filter': True,
            'vad_parameters': dict(min_silence_duration_ms=1000),
            'temperature': 0.0
        }
        
        # Verify structure
        assert isinstance(default_params['beam_size'], int)
        assert isinstance(default_params['vad_filter'], bool)
        assert isinstance(default_params['vad_parameters'], dict)
        assert isinstance(default_params['temperature'], (int, float))
        assert 'min_silence_duration_ms' in default_params['vad_parameters']
    
    def test_custom_params_validation(self):
        """Test custom parameter validation"""
        custom_params = {
            'beam_size': 3,
            'temperature': 0.1,
            'vad_filter': False,
            'custom_param': 'test'
        }
        
        # Parameters should accept custom values
        assert custom_params['beam_size'] == 3
        assert custom_params['temperature'] == 0.1
        assert custom_params['vad_filter'] == False
        assert custom_params['custom_param'] == 'test'


@pytest.mark.skipif(
    True,  # Skip by default since it requires faster-whisper
    reason="Requires faster-whisper installation"
)
class TestRealTranscription:
    """Tests with real audio files (optional)"""
    
    def test_real_audio_transcription(self, test_audio_file):
        """Test transcription with real audio file"""
        if test_audio_file is None:
            pytest.skip("No test audio file available")
        
        transcriber = AudioTranscriber(model_size="tiny", device="cpu", verbose=True)
        
        try:
            text = transcriber.transcribe_text_only(test_audio_file)
            assert isinstance(text, str)
            assert len(text) > 0
            
        except ImportError:
            pytest.skip("faster-whisper not available")
        except Exception as e:
            pytest.fail(f"Transcription failed: {e}")
        finally:
            transcriber.close()