# Docker Execution Instructions - Adobe Hackathon Round 1B

## Quick Start (Automated Processing)

### Process All Collections Automatically

```powershell
# Run the automated script to process all three collections
.\run_all_collections_docker.ps1
```

**Expected Output:**

- Collection 1: ~9-10 seconds
- Collection 2: ~14-15 seconds
- Collection 3: ~11-12 seconds
- Total time: ~55 seconds (including Docker overhead)

## Manual Docker Commands

### 1. Build the Pre-Warmed Docker Image

```powershell
# Build the optimized image with pre-warmed models
docker build -t adobe-hackathon-pipeline .
```

**Build Process:**

- Downloads and pre-warms `paraphrase-multilingual-MiniLM-L12-v2` model (278MB)
- Installs all dependencies and NLTK data
- Pre-warming takes ~60 seconds during build
- **Result**: Runtime model loading reduced from 55s to 6s

### 2. Process Individual Collections

#### Collection 1 (Travel Planner)

```powershell
docker run --rm -v "${PWD}/Collection 1:/app/input" -v "${PWD}/Collection 1:/app/output" adobe-hackathon-pipeline
```

#### Collection 2 (HR Professional)

```powershell
docker run --rm -v "${PWD}/Collection 2:/app/input" -v "${PWD}/Collection 2:/app/output" adobe-hackathon-pipeline
```

#### Collection 3 (Food Contractor)

```powershell
docker run --rm -v "${PWD}/Collection 3:/app/input" -v "${PWD}/Collection 3:/app/output" adobe-hackathon-pipeline
```

## Performance Verification

### Check Processing Times

Each collection will output:

```
Processing completed within time constraint: X.Xs
Performance constraint satisfied.
```

### Verify Output Files

After processing, check for `challenge1b_output.json` in each collection directory:

```powershell
# Verify all outputs exist
ls "Collection 1/challenge1b_output.json"
ls "Collection 2/challenge1b_output.json"
ls "Collection 3/challenge1b_output.json"
```

## Architecture Details

### Pre-Warmed Model Strategy

- **Build-time optimization**: Models loaded during Docker image construction
- **Runtime benefit**: Eliminates 55-second startup delay
- **Memory efficiency**: Singleton pattern prevents duplicate model loading
- **Performance result**: 9-15 second processing per collection

### Volume Mapping Strategy

- **Input mapping**: Collection PDFs mounted to `/app/input`
- **Output mapping**: Results written to `/app/output`
- **File preservation**: Results saved in respective collection directories
- **Docker cleanup**: `--rm` flag removes containers after execution

## Troubleshooting

### Image Build Issues

```powershell
# Clean build (if needed)
docker system prune -f
docker build --no-cache -t adobe-hackathon-pipeline .
```

### Performance Validation

```powershell
# Check image size
docker images adobe-hackathon-pipeline

# Verify pre-warming worked (should see model loading logs)
docker run --rm adobe-hackathon-pipeline python -c "from src.nlp.model_singleton import ModelSingleton; print('Pre-warming verified')"
```

### Collection Processing Verification

```powershell
# Test single collection with timing
Measure-Command { docker run --rm -v "${PWD}/Collection 1:/app/input" -v "${PWD}/Collection 1:/app/output" adobe-hackathon-pipeline }
```

## Expected Results

### Performance Metrics

- **Collection 1**: 9.27-9.38 seconds (Travel planning, 10 sections, 9 subsections)
- **Collection 2**: 14.19 seconds (HR forms, 49 sections, 9 subsections)
- **Collection 3**: 11.38 seconds (Food recipes, 16 sections, 4 subsections with vegetarian filtering)

### Quality Indicators

- **Intelligent filtering**: Collection 3 correctly filters 92 non-vegetarian recipes
- **Persona matching**: Content relevance scores > 0.15 threshold
- **Output structure**: JSON with sections, subsections, relevance scores, and metadata

### File Outputs

Each `challenge1b_output.json` contains:

# Docker Commands for Adobe Hackathon Round 1B

## Build the Docker Image

```powershell
docker build -t adobe-hackathon-pipeline .
```

## Run All Collections (Automated Script)

```powershell
.\run_all_collections_docker.ps1
```

## Run Individual Collections

```powershell
docker run --rm -v "${PWD}/Collection 1:/app/input" -v "${PWD}/Collection 1:/app/output" adobe-hackathon-pipeline
docker run --rm -v "${PWD}/Collection 2:/app/input" -v "${PWD}/Collection 2:/app/output" adobe-hackathon-pipeline
docker run --rm -v "${PWD}/Collection 3:/app/input" -v "${PWD}/Collection 3:/app/output" adobe-hackathon-pipeline
```

## Verify Output Files

```powershell
ls "Collection 1/challenge1b_output.json"
ls "Collection 2/challenge1b_output.json"
ls "Collection 3/challenge1b_output.json"
```
