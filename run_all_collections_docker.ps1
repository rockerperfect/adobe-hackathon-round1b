# Adobe Hackathon Round 1B - Docker Runner for All Collections
# This script processes all three collections using the containerized application
# Output files are saved in each collection's respective directory

Write-Host "Adobe Hackathon Round 1B - Processing All Collections via Docker" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green

# Collection 1: Travel Planner
Write-Host "`nProcessing Collection 1 (Travel Planner)..." -ForegroundColor Yellow
$start1 = Get-Date
docker run --rm -v "${PWD}/Collection 1:/app/input" -v "${PWD}/Collection 1:/app/output" adobe-hackathon-pipeline
$end1 = Get-Date
$duration1 = $end1 - $start1
Write-Host "Collection 1 completed in $($duration1.TotalSeconds) seconds" -ForegroundColor Green

# Collection 2: HR Professional  
Write-Host "`nProcessing Collection 2 (HR Professional)..." -ForegroundColor Yellow
$start2 = Get-Date
docker run --rm -v "${PWD}/Collection 2:/app/input" -v "${PWD}/Collection 2:/app/output" adobe-hackathon-pipeline
$end2 = Get-Date
$duration2 = $end2 - $start2
Write-Host "Collection 2 completed in $($duration2.TotalSeconds) seconds" -ForegroundColor Green

# Collection 3: Food Contractor
Write-Host "`nProcessing Collection 3 (Food Contractor)..." -ForegroundColor Yellow
$start3 = Get-Date
docker run --rm -v "${PWD}/Collection 3:/app/input" -v "${PWD}/Collection 3:/app/output" adobe-hackathon-pipeline
$end3 = Get-Date
$duration3 = $end3 - $start3
Write-Host "Collection 3 completed in $($duration3.TotalSeconds) seconds" -ForegroundColor Green

# Summary
$totalDuration = $duration1.TotalSeconds + $duration2.TotalSeconds + $duration3.TotalSeconds
Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "All Collections Processing Complete!" -ForegroundColor Green
Write-Host "Total processing time: $totalDuration seconds" -ForegroundColor Cyan
Write-Host "`nOutput files saved in:" -ForegroundColor White
Write-Host "  - Collection 1/challenge1b_output.json" -ForegroundColor Gray
Write-Host "  - Collection 2/challenge1b_output.json" -ForegroundColor Gray
Write-Host "  - Collection 3/challenge1b_output.json" -ForegroundColor Gray
Write-Host "`nDocker image: adobe-hackathon-pipeline" -ForegroundColor White
