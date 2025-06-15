"""
Unit tests for transcriber module
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile

from transcribe.transcriber import AudioTranscriber


class TestAudioTranscriberInit:
    """Test AudioTranscriber initialization"""
    
    def test_default_initialization(self):
        """Test default transcriber initialization"""
        transcriber = AudioTranscriber()
        assert transcriber.model_size == "base"
        assert transcriber.model is None
        # Device and compute_type are set by device detection
        assert transcriber.device in ["auto", "cpu", "cuda", "mps"]
        assert transcriber.compute_type in ["auto", "float32", "float16", "int8", "int8_float16"]
    
    def test_custom_initialization(self):
        """Test custom transcriber initialization"""
        transcriber = AudioTranscriber(
            model_size="tiny",
            device="cpu",
            compute_type="float32",
            debug=True,
            verbose=True
        )
        assert transcriber.model_size == "tiny"
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "float32"
        assert transcriber.logger.debug_mode
        assert transcriber.logger.verbose_mode
    
    @patch('transcribe.device_detection.get_optimal_device')
    @patch('transcribe.device_detection.get_optimal_compute_type')
    def test_device_detection_fallback(self, mock_compute_type, mock_device):
        """Test fallback when device detection is not available"""
        # Mock the device detection to succeed
        mock_device.return_value = "cpu"
        mock_compute_type.return_value = "float32"
        
        transcriber = AudioTranscriber(device="auto", compute_type="auto")
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "float32"


class TestModelManagement:
    """Test model loading and management"""
    
    def test_get_model_info_before_loading(self):
        """Test model info before model is loaded"""
        transcriber = AudioTranscriber(model_size="tiny", device="cpu")
        info = transcriber.get_model_info()
        
        assert info['model_size'] == "tiny"
        assert info['device'] == "cpu"
        assert info['is_loaded'] == False
    
    @patch('faster_whisper.WhisperModel')
    def test_model_initialization_success(self, mock_whisper_model):
        """Test successful model initialization"""
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        transcriber = AudioTranscriber(model_size="tiny", device="cpu")
        transcriber._initialize_model()
        
        assert transcriber.model is mock_model
        mock_whisper_model.assert_called_once_with("tiny", device="cpu", compute_type="float32")
    
    @patch('faster_whisper.WhisperModel')
    def test_model_initialization_fallback(self, mock_whisper_model):
        """Test model initialization fallback to CPU"""
        # First call fails, second succeeds
        mock_whisper_model.side_effect = [
            Exception("GPU not available"),
            MagicMock()
        ]
        
        transcriber = AudioTranscriber(model_size="tiny", device="mps")
        transcriber._initialize_model()
        
        # Should have attempted both original and fallback
        assert mock_whisper_model.call_count == 2
        # Second call should be fallback to CPU
        second_call = mock_whisper_model.call_args_list[1]
        assert second_call[1]['device'] == "cpu"
        assert second_call[1]['compute_type'] == "float32"
    
    def test_model_initialization_import_error(self):
        """Test model initialization with missing faster-whisper"""
        with patch('faster_whisper.WhisperModel', side_effect=ImportError()):
            transcriber = AudioTranscriber()
            
            with pytest.raises(ImportError, match="faster-whisper is not installed"):
                transcriber._initialize_model()
    
    def test_close_model(self):
        """Test model cleanup"""
        transcriber = AudioTranscriber()
        # Set a mock model
        transcriber.model = MagicMock()
        
        transcriber.close()
        assert transcriber.model is None


class TestTranscription:
    """Test transcription functionality"""
    
    @patch('transcribe.transcriber.get_audio_duration')
    def test_transcribe_text_only(self, mock_duration, temp_dir):
        """Test text-only transcription"""
        mock_duration.return_value = 10.0
        
        # Create test input file
        input_file = temp_dir / "test.mp3"
        input_file.write_bytes(b"fake audio")
        
        # Mock the transcription process
        with patch.object(AudioTranscriber, 'transcribe_file') as mock_transcribe:
            mock_transcribe.return_value = {
                'segments': [
                    {'text': 'First segment'},
                    {'text': 'Second segment'}
                ]
            }
            
            transcriber = AudioTranscriber(model_size="tiny")
            result = transcriber.transcribe_text_only(input_file)
            
            assert result == "First segment\nSecond segment"
            mock_transcribe.assert_called_once_with(input_file, transcription_params=None)
    
    @patch('transcribe.transcriber.get_audio_duration')
    @patch('builtins.open', new_callable=mock_open)
    def test_transcribe_file_with_output(self, mock_file, mock_duration, temp_dir):
        """Test file transcription with output"""
        mock_duration.return_value = 10.0
        
        # Create test files
        input_file = temp_dir / "input.mp3"
        output_file = temp_dir / "output.txt"
        input_file.write_bytes(b"fake audio")
        
        # Mock the model and transcription
        mock_model = MagicMock()
        mock_segment1 = MagicMock()
        mock_segment1.start = 0.0
        mock_segment1.end = 5.0
        mock_segment1.text = "First segment"
        
        mock_segment2 = MagicMock()
        mock_segment2.start = 5.0
        mock_segment2.end = 10.0
        mock_segment2.text = "Second segment"
        
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        # Fix f-string formatting issue with MagicMock
        type(mock_info.language_probability).__format__ = lambda self, format_spec: "0.99"
        
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
        
        transcriber = AudioTranscriber(model_size="tiny")
        transcriber.model = mock_model
        
        result = transcriber.transcribe_file(input_file, output_file)
        
        # Verify result structure
        assert 'segments' in result
        assert 'language' in result
        assert 'processing_time' in result
        assert result['language'] == "en"
        assert result['language_probability'] == 0.99
        assert len(result['segments']) == 2
        
        # Verify file was written
        mock_file.assert_called_with(output_file, 'w', encoding='utf-8')
    
    @patch('transcribe.transcriber.get_audio_duration')
    def test_transcribe_file_without_output(self, mock_duration, temp_dir):
        """Test file transcription without output file"""
        mock_duration.return_value = 5.0
        
        input_file = temp_dir / "input.mp3"
        input_file.write_bytes(b"fake audio")
        
        # Mock the model
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = "Test segment"
        
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        transcriber = AudioTranscriber(model_size="tiny")
        transcriber.model = mock_model
        
        result = transcriber.transcribe_file(input_file)
        
        # Should not have output_file in result
        assert 'output_file' not in result
        assert result['segment_count'] == 1
    
    def test_transcribe_file_failure(self, temp_dir):
        """Test transcription failure handling"""
        input_file = temp_dir / "input.mp3"
        input_file.write_bytes(b"fake audio")
        
        # Mock model that raises exception
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        
        transcriber = AudioTranscriber(model_size="tiny")
        transcriber.model = mock_model
        
        with pytest.raises(RuntimeError, match="Transcription failed"):
            transcriber.transcribe_file(input_file)


class TestTranscriptionParams:
    """Test transcription parameter handling"""
    
    @patch('transcribe.transcriber.get_audio_duration')
    def test_default_transcription_params(self, mock_duration, temp_dir):
        """Test default transcription parameters"""
        mock_duration.return_value = 5.0
        
        input_file = temp_dir / "input.mp3"
        input_file.write_bytes(b"fake audio")
        
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        
        transcriber = AudioTranscriber()
        transcriber.model = mock_model
        
        transcriber.transcribe_file(input_file)
        
        # Verify default parameters were used
        call_args = mock_model.transcribe.call_args
        assert call_args[1]['beam_size'] == 5
        assert call_args[1]['vad_filter'] == True
        assert call_args[1]['temperature'] == 0.0
    
    @patch('transcribe.transcriber.get_audio_duration') 
    def test_custom_transcription_params(self, mock_duration, temp_dir):
        """Test custom transcription parameters"""
        mock_duration.return_value = 5.0
        
        input_file = temp_dir / "input.mp3"
        input_file.write_bytes(b"fake audio")
        
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        
        transcriber = AudioTranscriber()
        transcriber.model = mock_model
        
        custom_params = {
            'beam_size': 3,
            'temperature': 0.1,
            'vad_filter': False
        }
        
        transcriber.transcribe_file(input_file, transcription_params=custom_params)
        
        # Verify custom parameters were used
        call_args = mock_model.transcribe.call_args
        assert call_args[1]['beam_size'] == 3
        assert call_args[1]['temperature'] == 0.1
        assert call_args[1]['vad_filter'] == False