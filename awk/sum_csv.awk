#!/usr/bin/awk -f
BEGIN {
    FS=","
    suma=0
    count=0
    print "Processing CSV file..."
}

NR == 1 {
    print "Header:", $0
    next  # Skip header row
}

{
    if ($2 ~ /^[0-9]+$/) {
        sum += $2
        count++
        printf "Row %d: Added %d to sum\n", NR, $2
    } else {
        printf "Row %d: Skipping non-numeric value '%s'\n", NR, $2
    }
}

END {
    print "\n=== Summary ==="
    print "Total rows processed:", NR
    print "Valid numeric rows:", count
    print "Total sum:", sum
    if (count > 0) {
        print "Average:", sum/count
    }
}