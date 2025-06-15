#!/usr/bin/env python3
"""
Tests for parallel processing functionality
"""
import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcribe import (
    get_audio_duration, should_use_parallel_processing, 
    get_optimal_thread_count, create_audio_chunks,
    transcribe_chunk, assemble_transcripts
)


class TestParallelProcessing(unittest.TestCase):
    """Test parallel processing functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_get_audio_duration_success(self, mock_run):
        """Test successful audio duration detection"""
        # Mock ffprobe output
        mock_run.return_value.stdout = "1800.5\n"
        mock_run.return_value.returncode = 0
        
        duration = get_audio_duration("test.mp3")
        
        self.assertEqual(duration, 1800.5)
        mock_run.assert_called_once()
        
    @patch('subprocess.run')
    def test_get_audio_duration_failure(self, mock_run):
        """Test audio duration detection failure"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ffprobe')
        
        with self.assertRaises(RuntimeError):
            get_audio_duration("nonexistent.mp3")
    
    def test_should_use_parallel_processing(self):
        """Test parallel processing decision logic"""
        # Test files under 30 minutes
        self.assertFalse(should_use_parallel_processing(1500))  # 25 minutes
        self.assertFalse(should_use_parallel_processing(1800))  # 30 minutes exactly
        
        # Test files over 30 minutes
        self.assertTrue(should_use_parallel_processing(1801))   # 30.02 minutes
        self.assertTrue(should_use_parallel_processing(3600))   # 60 minutes
    
    def test_get_optimal_thread_count_user_specified(self):
        """Test thread count with user specification"""
        self.assertEqual(get_optimal_thread_count(4), 4)
        self.assertEqual(get_optimal_thread_count(1), 1)
        
        with self.assertRaises(ValueError):
            get_optimal_thread_count(0)
    
    @patch('multiprocessing.cpu_count')
    def test_get_optimal_thread_count_auto(self, mock_cpu_count):
        """Test automatic thread count detection"""
        # Test with 8 cores
        mock_cpu_count.return_value = 8
        self.assertEqual(get_optimal_thread_count(None), 6)  # 75% of 8, max 8
        
        # Test with 16 cores
        mock_cpu_count.return_value = 16
        self.assertEqual(get_optimal_thread_count(None), 8)  # Max 8
        
        # Test with 2 cores
        mock_cpu_count.return_value = 2
        self.assertEqual(get_optimal_thread_count(None), 2)  # Min 2
    
    @patch('transcribe.get_audio_duration')
    @patch('subprocess.run')
    def test_create_audio_chunks(self, mock_run, mock_duration):
        """Test audio chunking functionality"""
        mock_duration.side_effect = [3000, 600, 600, 600, 600, 600]  # Total + 5 chunks
        mock_run.return_value.returncode = 0
        
        input_file = Path(self.temp_dir) / "test.mp3"
        input_file.touch()
        
        chunks = create_audio_chunks(input_file, 10, self.temp_dir)
        
        self.assertEqual(len(chunks), 5)  # 3000 seconds / 600 seconds per chunk
        
        # Verify chunk structure
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk['index'], i)
            self.assertEqual(chunk['start_time'], i * 600)
            self.assertTrue(chunk['file'].name.startswith('chunk_'))
            
    def test_transcribe_chunk_success(self):
        """Test successful chunk transcription"""
        # Mock chunk info
        chunk_info = {
            'index': 0,
            'file': Path(self.temp_dir) / "chunk_000.wav",
            'start_time': 0,
            'duration': 600
        }
        chunk_info['file'].touch()
        
        # Mock segments
        mock_segment1 = MagicMock()
        mock_segment1.start = 10.0
        mock_segment1.end = 15.0
        mock_segment1.text = "Hello world"
        
        mock_segment2 = MagicMock()
        mock_segment2.start = 20.0
        mock_segment2.end = 25.0
        mock_segment2.text = "Test segment"
        
        with patch('faster_whisper.WhisperModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95
            
            mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
            
            result = transcribe_chunk(chunk_info, "base", "float32")
            
            self.assertEqual(result['chunk_index'], 0)
            self.assertEqual(len(result['segments']), 2)
            self.assertEqual(result['language'], "en")
            
            # Check timestamp adjustment
            self.assertEqual(result['segments'][0]['start'], 10.0)  # 10.0 + 0
            self.assertEqual(result['segments'][0]['end'], 15.0)
            self.assertEqual(result['segments'][0]['text'], "Hello world")
    
    def test_transcribe_chunk_error(self):
        """Test chunk transcription error handling"""
        chunk_info = {
            'index': 0,
            'file': Path(self.temp_dir) / "chunk_000.wav",
            'start_time': 0,
            'duration': 600
        }
        
        with patch('faster_whisper.WhisperModel') as mock_model_class:
            mock_model_class.side_effect = Exception("Model loading failed")
            
            result = transcribe_chunk(chunk_info, "base", "float32")
            
            self.assertEqual(result['chunk_index'], 0)
            self.assertIn('error', result)
            self.assertEqual(result['segments'], [])
    
    def test_assemble_transcripts_success(self):
        """Test successful transcript assembly"""
        chunk_results = [
            {
                'chunk_index': 1,
                'segments': [
                    {'start': 600, 'end': 605, 'text': 'Second chunk'},
                    {'start': 610, 'end': 615, 'text': 'More text'}
                ],
                'language': 'en',
                'language_probability': 0.95
            },
            {
                'chunk_index': 0,
                'segments': [
                    {'start': 10, 'end': 15, 'text': 'First chunk'},
                    {'start': 20, 'end': 25, 'text': 'Early text'}
                ],
                'language': 'en',
                'language_probability': 0.93
            }
        ]
        
        segments, language_info = assemble_transcripts(chunk_results)
        
        # Should be sorted by start time
        self.assertEqual(len(segments), 4)
        self.assertEqual(segments[0]['text'], 'First chunk')
        self.assertEqual(segments[1]['text'], 'Early text')
        self.assertEqual(segments[2]['text'], 'Second chunk')
        self.assertEqual(segments[3]['text'], 'More text')
        
        # Language info from first chunk with segments
        self.assertEqual(language_info['language'], 'en')
        
    def test_assemble_transcripts_with_errors(self):
        """Test transcript assembly with chunk errors"""
        chunk_results = [
            {
                'chunk_index': 0,
                'segments': [{'start': 10, 'end': 15, 'text': 'Good chunk'}],
                'language': 'en',
                'language_probability': 0.95
            },
            {
                'chunk_index': 1,
                'error': 'Transcription failed',
                'segments': []
            }
        ]
        
        with self.assertRaises(RuntimeError) as context:
            assemble_transcripts(chunk_results)
        
        self.assertIn('Transcription errors', str(context.exception))
        self.assertIn('Chunk 1: Transcription failed', str(context.exception))


if __name__ == "__main__":
    unittest.main()