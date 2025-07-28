# Multi-Collection Processing Support

## Overview

The Adobe Hackathon Round 1B pipeline now supports processing multiple collections in the required hackathon format. This allows you to process all three challenge collections (Travel Planning, Adobe Acrobat Learning, Recipe Collection) in a single run.

## Collection Structure

Each collection must follow this directory structure:

```
Collection X/
├── PDFs/                       # Directory containing PDF documents
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── challenge1b_input.json      # Input configuration
└── challenge1b_output.json     # Generated analysis results (created by pipeline)
```

## Usage

### Multi-Collection Mode (Recommended for Hackathon)

Process all collections at once:

```bash
python main.py --collections_dir /path/to/collections/
```

Example with typical hackathon structure:

```bash
python main.py --collections_dir ./
```

This will process all directories starting with "Collection" in the specified directory.

### Single Collection Mode (Backward Compatible)

Process one collection at a time:

```bash
python main.py --input "Collection 1/challenge1b_input.json" --output "Collection 1/challenge1b_output.json"
```

## Expected Directory Layout for Hackathon

```
your-submission/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                       # South of France guides
│   │   ├── South of France - Cities.pdf
│   │   ├── South of France - Cuisine.pdf
│   │   └── ...
│   ├── challenge1b_input.json      # Travel planner configuration
│   └── challenge1b_output.json     # Generated results
├── Collection 2/                    # Adobe Acrobat Learning
│   ├── PDFs/                       # Acrobat tutorials
│   │   ├── acrobat_guide1.pdf
│   │   └── ...
│   ├── challenge1b_input.json      # HR professional configuration
│   └── challenge1b_output.json     # Generated results
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                       # Cooking guides
│   │   ├── recipe_guide1.pdf
│   │   └── ...
│   ├── challenge1b_input.json      # Food contractor configuration
│   └── challenge1b_output.json     # Generated results
├── main.py                          # Pipeline entrypoint
├── config/                          # Configuration files
├── src/                            # Source code
└── requirements.txt                # Dependencies
```

## Input JSON Format

Each `challenge1b_input.json` must contain:

```json
{
  "challenge_info": {
    "test_case_name": "travel_planner",
    "challenge_id": "round_1b_002"
  },
  "documents": [
    { "filename": "document1.pdf" },
    { "filename": "document2.pdf" }
  ],
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  }
}
```

## Output JSON Format

Each `challenge1b_output.json` will contain:

```json
{
  "metadata": {
    "input_documents": ["document1.pdf", "document2.pdf"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
    "processing_timestamp": "2025-07-28T04:00:26.886"
  },
  "extracted_sections": [
    {
      "document": "document1.pdf",
      "section_title": "Travel Tips",
      "importance_rank": 1,
      "page_number": 2
    }
  ],
  "subsection_analysis": [
    {
      "document": "document1.pdf",
      "refined_text": "Detailed travel recommendations...",
      "page_number": 2
    }
  ]
}
```

## Features

- **Dynamic Path Discovery**: Automatically finds and processes all collections
- **Error Handling**: Continues processing other collections if one fails
- **Progress Logging**: Detailed logging for each collection
- **Performance Monitoring**: Tracks processing time per collection
- **Validation**: Ensures output format matches expected structure
- **Fallback Output**: Creates minimal output if processing fails

## Performance

- **Collection 1 (Travel)**: ~33 seconds for 7 PDFs, 43 sections, 30 subsections
- **Collection 2 (HR)**: ~25 seconds for 1 PDF, 6 sections, 11 subsections
- **Total**: Processes all collections within hackathon time constraints

## Testing

Test the multi-collection support:

```bash
python test_collections.py
```

This creates sample collections and validates the processing functionality.

## Error Handling

If a collection fails:

- Error is logged but processing continues with other collections
- Fallback output JSON is created with empty sections
- Final summary shows successful vs failed collections
- Exit code indicates overall success/failure

## Backward Compatibility

The pipeline maintains full backward compatibility:

- Single collection mode still works with `--input` and `--output`
- All existing configuration and features preserved
- No breaking changes to existing workflows

## Notes

- PDFs must be in the `PDFs/` subdirectory of each collection
- Collection directories must start with "Collection" (case-insensitive)
- All paths are resolved dynamically (no hardcoding)
- Unicode output is supported for international content
- ML models are loaded once per collection for efficiency
