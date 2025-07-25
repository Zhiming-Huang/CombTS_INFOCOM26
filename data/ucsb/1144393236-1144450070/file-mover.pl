#!/usr/bin/perl
$dir = $ARGV[0];
$start = $ARGV[1];
$end = $ARGV[2];

@files = `ls $dir`;

foreach $file (@files) {
	chomp $file;
	if ($file =~ /neighbortable-(\d+)/) {
		$t = $1;
		if ($t > $start && $t < $end) {
			`mv $dir/$file .`;
		}
	}
}
