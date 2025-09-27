Param(
    [Parameter(Mandatory = $true)]
    [string]$FolderPath
)

# Check that the folder exists
if (-Not (Test-Path $FolderPath)) {
    Write-Error "The specified folder path does not exist: $FolderPath"
    exit 1
}

# Get all *.pcap files from the directory, sorted by name
$files = Get-ChildItem -Path $FolderPath -Filter *.pcap | Sort-Object Name

# Optional: warn if the number of files is not 100
if ($files.Count -ne 100) {
    Write-Warning "Expected 100 pcap files but found $($files.Count) files in $FolderPath."
}

# Rename each file to a sequential two-digit number with .pcap extension
$count = 0
foreach ($file in $files) {
    # Format the counter as a two-digit number (00, 01, …, 99)
    $newName = "{0:D2}.pcap" -f $count

    # Build the full path for the new name
    $newPath = Join-Path -Path $FolderPath -ChildPath $newName

    # Rename the file
    Rename-Item -Path $file.FullName -NewName $newName

    $count++
}
