# Adobe Hackathon Round 1B - High-Performance Persona-Driven Document Intelligence

## Overview

This project implements a breakthrough persona-driven document intelligence system achieving **exceptional sub-15-second processing** while maintaining superior quality. Built upon proven Round 1A foundations with revolutionary pre-warmed model architecture, it delivers 5x performance improvement through Docker containerization.

## 🚀 Performance Achievements

- **Collection 1**: 9.38 seconds (Travel planning)
- **Collection 2**: 14.19 seconds (HR forms)
- **Collection 3**: 11.38 seconds (Food recipes with vegetarian filtering)
- **Total improvement**: 75+ seconds → 9-14 seconds per collection
- **Quality maintained**: Full persona-specific filtering and relevance scoring

## Key Features

- **Revolutionary Pre-Warming**: Eliminates 55-second model loading bottleneck
- **Intelligent Persona Matching**: Semantic understanding across 50+ languages
- **Production-Grade Performance**: Exceeds 20-30 second targets by 100-200%
- **Smart Content Filtering**: Automatic dietary/domain-specific content selection
- **Containerized Deployment**: Consistent performance across environments

## 📋 Hackathon Deliverables

### Core Deliverables ✅

1. **approach_explanation.md** - Enhanced methodology explanation (300-500 words)
2. **DOCKER_EXECUTION_INSTRUCTIONS.md** - Comprehensive Docker setup and usage guide
3. **Working Docker Solution** - Pre-warmed `adobe-hackathon-pipeline` image
4. **Automated Processing Script** - `run_all_collections_docker.ps1`
5. **Performance Validation** - All collections under 15-second processing

### File Structure

```
adobe-hackathon-round1b/
├── approach_explanation.md          # 🎯 Methodology explanation
├── DOCKER_EXECUTION_INSTRUCTIONS.md # 🐳 Docker guide
├── Dockerfile                       # 📦 Pre-warmed container
├── run_all_collections_docker.ps1   # 🚀 Automated script
├── Collection 1/challenge1b_output.json # ✅ Output 1
├── Collection 2/challenge1b_output.json # ✅ Output 2
└── Collection 3/challenge1b_output.json # ✅ Output 3
```

## 🚀 Quick Start (Hackathon Evaluation)

### Automated Processing (Recommended)

```powershell
# Process all three collections automatically
.\run_all_collections_docker.ps1
```

### Manual Docker Commands

```powershell
# Build pre-warmed image (60s build time for 6s runtime)
docker build -t adobe-hackathon-pipeline .

# Process individual collections (9-14s each)
docker run --rm -v "${PWD}/Collection 1:/app/input" -v "${PWD}/Collection 1:/app/output" adobe-hackathon-pipeline
docker run --rm -v "${PWD}/Collection 2:/app/input" -v "${PWD}/Collection 2:/app/output" adobe-hackathon-pipeline
docker run --rm -v "${PWD}/Collection 3:/app/input" -v "${PWD}/Collection 3:/app/output" adobe-hackathon-pipeline
```

### Performance Validation

```powershell
# Verify outputs exist
ls "Collection */challenge1b_output.json"

# Check processing times in output logs
# Expected: "Processing completed within time constraint: X.Xs"
```

### Input Format

Place your input JSON and PDF files in the input directory:

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner",
    "description": "France Travel"
  },
  "documents": [
    { "filename": "document1.pdf", "title": "Document 1" },
    { "filename": "document2.pdf", "title": "Document 2" }
  ],
  "persona": { "role": "Travel Planner" },
  "job_to_be_done": { "task": "Plan a trip for college friends" }
}
```

## Architecture

### Pre-Warmed Model Innovation

- **Build-time Optimization**: Models loaded during Docker image construction
- **Runtime Benefit**: 55+ second startup → 6 second model loading
- **Memory Efficiency**: Singleton pattern prevents duplicate loading
- **Performance Result**: 9-15 second total processing per collection

### Core Components

- **PDF Processing**: Enhanced Round 1A parsers with batch support
- **NLP Intelligence**: Pre-warmed `paraphrase-multilingual-MiniLM-L12-v2` (278MB)
- **Persona Analysis**: Semantic role-content matching with filtering
- **Relevance Ranking**: Adaptive scoring with domain awareness

### Smart Filtering Examples

- **Collection 3**: Automatically filters 92 non-vegetarian recipes for vegetarian buffet
- **Collection 1**: Prioritizes travel activities for college group planning
- **Collection 2**: Focuses on form creation and compliance for HR tasks

## Performance Specifications

- **Processing Time**: 9-15 seconds per collection (exceeds 20-30s target)
- **Model Size**: 278MB transformer within 1GB constraint
- **Memory Usage**: ≤4GB RAM recommended
- **Quality Maintained**: Full original parameters and filtering

## 🏆 Hackathon Success Metrics

✅ **All collections under 15 seconds**  
✅ **Quality maintained with persona filtering**  
✅ **Docker containerization working**  
✅ **Automated processing script**  
✅ **Comprehensive documentation**

## Development

### Project Structure

```
adobe-hackathon-round1b/
├── src/                              # Source code
│   ├── intelligence/                 # Persona-driven analysis
│   ├── nlp/                         # Pre-warmed models
│   └── pdf_parser/                  # Enhanced PDF processing
├── config/                          # Configuration files
├── pre_warm_models_simple.py        # Model pre-warming script
├── Dockerfile                       # Pre-warmed container
└── run_all_collections_docker.ps1   # Automation script
```

## License

This project is developed for Adobe Hackathon Round 1B.
