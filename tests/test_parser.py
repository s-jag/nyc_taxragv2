"""Tests for the LegalParser class."""

import tempfile
from pathlib import Path

import pytest

from app.ingest import LegalParser, ParentDocument


class TestParentDocument:
    """Tests for ParentDocument model."""

    def test_parent_document_creation(self) -> None:
        """Test basic ParentDocument instantiation."""
        doc = ParentDocument(
            id="test-uuid",
            section_id="11-121",
            text="Test content",
            source="NYC Tax Law"
        )
        assert doc.id == "test-uuid"
        assert doc.section_id == "11-121"
        assert doc.text == "Test content"
        assert doc.source == "NYC Tax Law"

    def test_parent_document_default_source(self) -> None:
        """Test that source defaults to 'NYC Tax Law'."""
        doc = ParentDocument(
            id="test-uuid",
            section_id="11-121",
            text="Test content"
        )
        assert doc.source == "NYC Tax Law"


class TestLegalParser:
    """Tests for LegalParser class."""

    @pytest.fixture
    def parser(self) -> LegalParser:
        """Create a LegalParser instance for testing."""
        return LegalParser()

    @pytest.fixture
    def sample_legal_text(self) -> str:
        """Sample legal text with multiple sections."""
        return """§ 11-121 City collector; daily statements and accounts.
   a.   The city collector or the deputy collector in each borough office of
the city collector shall enter upon accounts, to be maintained in each such
office for each parcel of property, the payment of taxes, assessments, sewer
rents or water rents thereon, the amount therefor, and the date when paid.
   b.   At close of office hours each day, the city collector shall render to
the commissioner of finance a statement of the sums so received.
§ 11-201 Assessments on real property; general powers of finance department.
The commissioner of finance shall be charged generally with the duty and
responsibility of assessing all real property subject to taxation within the
city.
§ 11-207.1 Information related to estimate of assessed valuation.
   a.   Not later than the fifteenth day of February, the commissioner of
finance shall submit the following information."""

    def test_load_data_parses_sections(
        self, parser: LegalParser, sample_legal_text: str
    ) -> None:
        """Test that load_data correctly parses multiple sections."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(sample_legal_text)
            temp_path = f.name

        try:
            documents = parser.load_data(temp_path)
            assert len(documents) == 3
        finally:
            Path(temp_path).unlink()

    def test_section_11_121_correctly_identified(
        self, parser: LegalParser, sample_legal_text: str
    ) -> None:
        """Test that section 11-121 is correctly identified."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(sample_legal_text)
            temp_path = f.name

        try:
            documents = parser.load_data(temp_path)
            section_ids = [doc.section_id for doc in documents]
            assert "11-121" in section_ids
        finally:
            Path(temp_path).unlink()

    def test_full_body_text_captured(
        self, parser: LegalParser, sample_legal_text: str
    ) -> None:
        """Test that full section body text is captured (not truncated)."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(sample_legal_text)
            temp_path = f.name

        try:
            documents = parser.load_data(temp_path)
            doc_11_121 = next(d for d in documents if d.section_id == "11-121")

            # Verify key content is present
            assert "City collector" in doc_11_121.text
            assert "daily statements and accounts" in doc_11_121.text
            assert "city collector shall render" in doc_11_121.text
            assert "commissioner of finance" in doc_11_121.text
        finally:
            Path(temp_path).unlink()

    def test_whitespace_cleaned(
        self, parser: LegalParser, sample_legal_text: str
    ) -> None:
        """Test that whitespace is properly cleaned."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(sample_legal_text)
            temp_path = f.name

        try:
            documents = parser.load_data(temp_path)

            for doc in documents:
                # No leading/trailing whitespace
                assert doc.text == doc.text.strip()
                # No multiple consecutive spaces
                assert "  " not in doc.text
                # No newlines
                assert "\n" not in doc.text
        finally:
            Path(temp_path).unlink()

    def test_section_with_decimal_id(self, parser: LegalParser) -> None:
        """Test that section IDs with decimals are correctly parsed."""
        text = """§ 11-207.1 Information related to valuation.
Some content here about valuation."""

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(text)
            temp_path = f.name

        try:
            documents = parser.load_data(temp_path)
            assert len(documents) == 1
            assert documents[0].section_id == "11-207.1"
        finally:
            Path(temp_path).unlink()

    def test_uuid_generation(
        self, parser: LegalParser, sample_legal_text: str
    ) -> None:
        """Test that each document gets a unique UUID."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(sample_legal_text)
            temp_path = f.name

        try:
            documents = parser.load_data(temp_path)
            ids = [doc.id for doc in documents]

            # All IDs should be unique
            assert len(ids) == len(set(ids))

            # All IDs should be valid UUID format
            for doc_id in ids:
                assert len(doc_id) == 36  # UUID format: 8-4-4-4-12
                assert doc_id.count('-') == 4
        finally:
            Path(temp_path).unlink()

    def test_source_field(
        self, parser: LegalParser, sample_legal_text: str
    ) -> None:
        """Test that all documents have correct source field."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(sample_legal_text)
            temp_path = f.name

        try:
            documents = parser.load_data(temp_path)
            for doc in documents:
                assert doc.source == "NYC Tax Law"
        finally:
            Path(temp_path).unlink()


class TestLegalParserWithRealData:
    """Integration tests using actual tax law data."""

    @pytest.fixture
    def parser(self) -> LegalParser:
        """Create a LegalParser instance for testing."""
        return LegalParser()

    @pytest.fixture
    def tax_law_path(self) -> Path:
        """Path to the actual tax law file."""
        return Path(__file__).parent.parent / "data" / "tax_law.txt"

    def test_parse_real_tax_law_file(
        self, parser: LegalParser, tax_law_path: Path
    ) -> None:
        """Test parsing the actual tax law file."""
        if not tax_law_path.exists():
            pytest.skip("Tax law file not found")

        documents = parser.load_data(tax_law_path)

        # Should have 819 sections based on analysis
        assert len(documents) == 819

    def test_real_file_section_11_121_exists(
        self, parser: LegalParser, tax_law_path: Path
    ) -> None:
        """Test that section 11-121 exists in real file."""
        if not tax_law_path.exists():
            pytest.skip("Tax law file not found")

        documents = parser.load_data(tax_law_path)
        section_ids = [doc.section_id for doc in documents]

        assert "11-121" in section_ids

    def test_real_file_section_11_121_content(
        self, parser: LegalParser, tax_law_path: Path
    ) -> None:
        """Test that section 11-121 has correct content."""
        if not tax_law_path.exists():
            pytest.skip("Tax law file not found")

        documents = parser.load_data(tax_law_path)
        doc_11_121 = next(d for d in documents if d.section_id == "11-121")

        assert "City collector" in doc_11_121.text
        assert "daily statements and accounts" in doc_11_121.text
