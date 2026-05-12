"""Type stubs for the CISV native extension module."""

from typing import List, Tuple
import numpy as np
import numpy.typing as npt

def parse_file(
    path: str,
    delimiter: str = ",",
    quote: str = '"',
    trim: bool = False,
    skip_empty_lines: bool = False,
    escape: str = "",
    comment: str = "",
    relaxed: bool = False,
    skip_lines_with_error: bool = False,
    max_row_size: int = 0,
    from_line: int = 1,
    to_line: int = 0,
) -> List[List[str]]:
    """
    Parse a CSV file and return all rows as a list of lists.

    Args:
        path: Path to the CSV file
        delimiter: Field delimiter character (default: ',')
        quote: Quote character (default: '"')
        trim: Whether to trim whitespace from fields
        skip_empty_lines: Whether to skip empty lines
        escape: Optional escape character, empty string for RFC doubled quotes
        comment: Optional comment prefix character
        relaxed: Keep parsing through relaxed parse errors when core supports it
        skip_lines_with_error: Skip malformed lines when core supports it
        max_row_size: Maximum row size, 0 for default/adaptive core behavior
        from_line: First 1-based line to parse
        to_line: Last 1-based line to parse, 0 for no upper bound

    Returns:
        List of rows, where each row is a list of field values
    """
    ...

def parse_string(
    data: str,
    delimiter: str = ",",
    quote: str = '"',
    trim: bool = False,
    skip_empty_lines: bool = False,
    escape: str = "",
    comment: str = "",
    relaxed: bool = False,
    skip_lines_with_error: bool = False,
    max_row_size: int = 0,
    from_line: int = 1,
    to_line: int = 0,
) -> List[List[str]]:
    """
    Parse a CSV string and return all rows as a list of lists.

    Args:
        data: CSV content as a string
        delimiter: Field delimiter character (default: ',')
        quote: Quote character (default: '"')
        trim: Whether to trim whitespace from fields
        skip_empty_lines: Whether to skip empty lines
        escape: Optional escape character, empty string for RFC doubled quotes
        comment: Optional comment prefix character
        relaxed: Keep parsing through relaxed parse errors when core supports it
        skip_lines_with_error: Skip malformed lines when core supports it
        max_row_size: Maximum row size, 0 for default/adaptive core behavior
        from_line: First 1-based line to parse
        to_line: Last 1-based line to parse, 0 for no upper bound

    Returns:
        List of rows, where each row is a list of field values
    """
    ...

def parse_file_parallel(
    path: str,
    num_threads: int = 0,
    delimiter: str = ",",
    quote: str = '"',
    trim: bool = False,
    skip_empty_lines: bool = False,
    escape: str = "",
    comment: str = "",
    relaxed: bool = False,
    skip_lines_with_error: bool = False,
    max_row_size: int = 0,
    from_line: int = 1,
    to_line: int = 0,
) -> List[List[str]]:
    """
    Parse a CSV file using multiple threads for maximum performance.

    Args:
        path: Path to the CSV file
        num_threads: Number of threads to use (0 = auto-detect)
        delimiter: Field delimiter character (default: ',')
        quote: Quote character (default: '"')
        trim: Whether to trim whitespace from fields
        skip_empty_lines: Whether to skip empty lines
        escape: Optional escape character, empty string for RFC doubled quotes
        comment: Optional comment prefix character
        relaxed: Keep parsing through relaxed parse errors when core supports it
        skip_lines_with_error: Skip malformed lines when core supports it
        max_row_size: Maximum row size, 0 for default/adaptive core behavior
        from_line: First 1-based line to parse
        to_line: Last 1-based line to parse, 0 for no upper bound

    Returns:
        List of rows, where each row is a list of field values
    """
    ...

def parse_file_raw(
    path: str,
    num_threads: int = 0,
    delimiter: str = ",",
    quote: str = '"',
    trim: bool = False,
    skip_empty_lines: bool = False,
    escape: str = "",
    comment: str = "",
    relaxed: bool = False,
    skip_lines_with_error: bool = False,
    max_row_size: int = 0,
    from_line: int = 1,
    to_line: int = 0,
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.uint64]]:
    """
    Ultra-fast parallel parsing that returns raw numpy arrays.

    Returns (data, field_offsets, field_lengths, row_offsets) where:
      - data: uint8 array of all field bytes
      - field_offsets: uint64 array of byte offsets for each field
      - field_lengths: uint32 array of byte lengths for each field
      - row_offsets: uint64 array of field indices for each row
    """
    ...

def count_rows(
    path: str,
    delimiter: str = ",",
    quote: str = '"',
    escape: str = "",
    comment: str = "",
    trim: bool = False,
    skip_empty_lines: bool = False,
    relaxed: bool = False,
    skip_lines_with_error: bool = False,
    max_row_size: int = 0,
    from_line: int = 1,
    to_line: int = 0,
) -> int:
    """
    Count the number of rows in a CSV file without full parsing.

    This uses core fast paths for simple, quoted, and semantic row-control counts.

    Args:
        path: Path to the CSV file
        delimiter: Field delimiter character
        quote: Quote character
        escape: Optional escape character, empty string for RFC doubled quotes
        comment: Optional comment prefix character
        trim: Trim unquoted fields before comment checks
        skip_empty_lines: Skip physically empty rows
        relaxed: Keep counting through relaxed parse errors when core supports it
        skip_lines_with_error: Skip malformed lines when core supports it
        max_row_size: Maximum row size, 0 for default/adaptive core behavior
        from_line: First 1-based line to count
        to_line: Last 1-based line to count, 0 for no upper bound

    Returns:
        Number of rows in the file
    """
    ...

class CisvIterator:
    """
    Row-by-row iterator for streaming CSV parsing.

    Provides fgetcsv-style iteration with minimal memory footprint.
    Supports early exit - no wasted work when breaking mid-file.

    Example:
        with CisvIterator('/path/to/file.csv') as reader:
            for row in reader:
                print(row)  # List[str]
                if row[0] == 'stop':
                    break  # Early exit - no wasted work
    """

    def __init__(
        self,
        path: str,
        delimiter: str = ",",
        quote: str = '"',
        trim: bool = False,
        skip_empty_lines: bool = False,
    ) -> None:
        """
        Create a new CSV iterator.

        Args:
            path: Path to the CSV file
            delimiter: Field delimiter character (default: ',')
            quote: Quote character (default: '"')
            trim: Whether to trim whitespace from fields
            skip_empty_lines: Whether to skip empty lines
        """
        ...

    def next(self) -> List[str] | None:
        """
        Get the next row as a list of strings, or None if at end of file.

        Returns:
            List of field values for the next row, or None if EOF
        """
        ...

    def close(self) -> None:
        """Close the iterator and release resources."""
        ...

    @property
    def closed(self) -> bool:
        """Whether the iterator has been closed."""
        ...

    def __iter__(self) -> "CisvIterator":
        """Return the iterator object itself."""
        ...

    def __next__(self) -> List[str]:
        """
        Get the next row as a list of strings.

        Raises:
            StopIteration: When end of file is reached
        """
        ...

    def __enter__(self) -> "CisvIterator":
        """Enter the context manager."""
        ...

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool:
        """Exit the context manager and close the iterator."""
        ...
