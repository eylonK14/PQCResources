Param(
    [Parameter(Mandatory = $true)]
    [string]$FolderPath, 
	[Parameter(Mandatory = $true)]
	[Int]$Amount
)

# Check that the folder exists
if (-Not (Test-Path $FolderPath)) {
    Write-Error "The specified folder path does not exist: $FolderPath"
    exit 1
}

# Check that the folder exists
if ($Amount -le 0) {
    Write-Error "Invalid Amount: Must be greater that 0!"
    exit 1
}

# Get all *.pcap files from the directory, sorted by name
$files = Get-ChildItem -Path $FolderPath -Filter *.pcap | Sort-Object Name

# Optional: warn if the number of files is not 200
if ($files.Count -ne $Amount) {
    Write-Warning "Expected $($Amount) pcap files but found $($files.Count) files in $FolderPath."
}

# Rename each file to a sequential three-digit number with .pcap extension
$count = 0
foreach ($file in $files) {
    # Format the counter as a three-digit number (000, 001, …, 199)
    $newName = "{0:D3}.pcap" -f $count

    # Build the full path for the new name
    $newPath = Join-Path -Path $FolderPath -ChildPath $newName

    # Rename the file
    Rename-Item -Path $file.FullName -NewName $newName

    $count++
}
