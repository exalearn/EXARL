#!/usr/bin/perl
use strict;
use Cwd qw(cwd);
use File::Basename;
use FindBin;
use lib "$FindBin::RealBin";
require SingleRun;

# These are used to submit and throttle commands sent to slurm
my $username = $ENV{LOGNAME} || $ENV{USER} || getpwuid($<);
my $partition  = shift(@ARGV);
print("Username: $username Slurm Partition: $partition\n");
SingleRun::setPartition("$partition");
SingleRun::setUser("$username");

my $command  = shift(@ARGV);
SingleRun::throttleCommand($command);
