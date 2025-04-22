"""
Tests for the PDF generator module.
"""
import os
import base64
import tempfile
import unittest
from unittest import mock
from io import BytesIO
from datetime import datetime

# Mock dependencies to avoid requiring external libraries
mock_imports = {
    'weasyprint': mock.MagicMock(),
    'pdfkit': mock.MagicMock(),
}

with mock.patch.dict('sys.modules', mock_imports):
    from src.pdf_generator import PDFGenerator, generate_executive_pdf, generate_from_streamlit_ui


class TestPDFGenerator(unittest.TestCase):
    """Tests for the PDFGenerator class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a temporary directory for storing reports
        self.temp_dir = tempfile.mkdtemp()
        self.pdf_generator = PDFGenerator(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up the test case."""
        # Clean up the temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    @mock.patch('pdfkit.from_string')
    def test_generate_from_html(self, mock_from_string):
        """Test generating a PDF from HTML content."""
        html_content = "<html><body><h1>Test PDF</h1></body></html>"
        mock_from_string.return_value = True
        
        filename = "test_pdf.pdf"
        filepath = self.pdf_generator.generate_from_html(html_content, filename)
        
        self.assertTrue(filepath.endswith(filename))
        self.assertTrue(filepath.startswith(self.temp_dir))
        mock_from_string.assert_called_once()

    @mock.patch('pdfkit.from_string')
    def test_generate_from_html_with_default_filename(self, mock_from_string):
        """Test generating a PDF with default filename."""
        html_content = "<html><body><h1>Test PDF</h1></body></html>"
        mock_from_string.return_value = True
        
        filepath = self.pdf_generator.generate_from_html(html_content)
        
        self.assertTrue(filepath.endswith('.pdf'))
        self.assertTrue(filepath.startswith(self.temp_dir))
        mock_from_string.assert_called_once()

    @mock.patch('pdfkit.from_string', side_effect=Exception("Test error"))
    @mock.patch('weasyprint.HTML')
    def test_generate_from_html_fallback(self, mock_html, mock_from_string):
        """Test fallback to WeasyPrint when pdfkit fails."""
        html_content = "<html><body><h1>Test PDF</h1></body></html>"
        
        # Configure the mock
        mock_html_instance = mock.MagicMock()
        mock_html.return_value = mock_html_instance
        mock_html_instance.write_pdf.return_value = True
        
        filename = "test_pdf_fallback.pdf"
        filepath = self.pdf_generator.generate_from_html(html_content, filename)
        
        self.assertTrue(filepath.endswith(filename))
        self.assertTrue(filepath.startswith(self.temp_dir))
        mock_from_string.assert_called_once()
        mock_html.assert_called_once_with(string=html_content)
        mock_html_instance.write_pdf.assert_called_once()
    
    @mock.patch('pdfkit.from_file')
    def test_generate_from_streamlit(self, mock_from_file):
        """Test generating a PDF from a Streamlit component."""
        # Mock the report function
        def mock_report_func(params):
            return None
        
        mock_from_file.return_value = True
        
        # Patch the internal method that generates HTML
        with mock.patch.object(self.pdf_generator, '_run_streamlit_to_html') as mock_run:
            mock_run.return_value = "<html><body><h1>Test Streamlit</h1></body></html>"
            
            filename = "test_streamlit.pdf"
            filepath = self.pdf_generator.generate_from_streamlit(
                report_func=mock_report_func,
                params={"test": "value"},
                filename=filename
            )
            
            mock_run.assert_called_once_with(mock_report_func, {"test": "value"})
            self.assertTrue(filepath.endswith(filename))
            self.assertTrue(filepath.startswith(self.temp_dir))
            mock_from_file.assert_called_once()
    
    @mock.patch('pdfkit.from_file', side_effect=Exception("Test error"))
    @mock.patch('weasyprint.HTML')
    def test_generate_from_streamlit_fallback(self, mock_html, mock_from_file):
        """Test fallback to WeasyPrint when pdfkit fails for Streamlit PDFs."""
        # Mock the report function
        def mock_report_func(params):
            return None
        
        # Configure the mock
        mock_html_instance = mock.MagicMock()
        mock_html.return_value = mock_html_instance
        mock_html_instance.write_pdf.return_value = True
        
        # Patch the internal method that generates HTML
        with mock.patch.object(self.pdf_generator, '_run_streamlit_to_html') as mock_run:
            mock_run.return_value = "<html><body><h1>Test Streamlit Fallback</h1></body></html>"
            
            filename = "test_streamlit_fallback.pdf"
            filepath = self.pdf_generator.generate_from_streamlit(
                report_func=mock_report_func,
                filename=filename
            )
            
            mock_run.assert_called_once()
            self.assertTrue(filepath.endswith(filename))
            self.assertTrue(filepath.startswith(self.temp_dir))
            mock_from_file.assert_called_once()
            mock_html.assert_called_once()
            mock_html_instance.write_pdf.assert_called_once()
    
    def test_run_streamlit_to_html(self):
        """Test running a Streamlit function to generate HTML."""
        # Mock the report function
        def mock_report_func(params):
            return None
        
        html = self.pdf_generator._run_streamlit_to_html(
            report_func=mock_report_func,
            params={"test": "value"}
        )
        
        self.assertIsInstance(html, str)
        self.assertTrue(html.startswith('<!DOCTYPE html>'))
        self.assertIn('Watchdog AI Executive Report', html)
        self.assertIn('Test Insights', html) # Check template rendering

    def test_generate_sample_charts(self):
        """Test generating sample charts for the template."""
        # Test sales chart
        sales_chart = self.pdf_generator._generate_sample_sales_chart()
        self.assertIsInstance(sales_chart, str)
        self.assertTrue(base64.b64decode(sales_chart))  # Valid base64
        
        # Test lead chart
        lead_chart = self.pdf_generator._generate_sample_lead_chart()
        self.assertIsInstance(lead_chart, str)
        self.assertTrue(base64.b64decode(lead_chart))  # Valid base64
        
        # Test inventory chart
        inventory_chart = self.pdf_generator._generate_sample_inventory_chart()
        self.assertIsInstance(inventory_chart, str)
        self.assertTrue(base64.b64decode(inventory_chart))  # Valid base64


@mock.patch('src.pdf_generator.PDFGenerator')
def test_generate_executive_pdf(mock_generator_class):
    """Test the generate_executive_pdf function."""
    # Set up the mock
    mock_generator = mock.MagicMock()
    mock_generator_class.return_value = mock_generator
    mock_generator.generate_from_streamlit.return_value = '/test/path/report.pdf'
    
    # Test with default parameters
    filepath = generate_executive_pdf()
    
    mock_generator_class.assert_called_once()
    mock_generator.generate_from_streamlit.assert_called_once()
    assert isinstance(filepath, str)


@mock.patch('src.pdf_generator.PDFGenerator')
def test_generate_from_streamlit_ui(mock_generator_class):
    """Test the generate_from_streamlit_ui function."""
    # Set up the mock
    mock_generator = mock.MagicMock()
    mock_generator_class.return_value = mock_generator
    mock_generator.generate_from_streamlit.return_value = '/test/path/ui_report.pdf'
    
    # Mock UI function
    def mock_ui_func(test_param):
        return None
    
    # Test the function
    filepath = generate_from_streamlit_ui(
        block=mock_ui_func,
        filename="ui_test.pdf",
        test_param="test value"
    )
    
    mock_generator_class.assert_called_once()
    mock_generator.generate_from_streamlit.assert_called_once()
    assert isinstance(filepath, str)