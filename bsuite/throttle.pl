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
my $command  = shift(@ARGV);
my $count = shift(@ARGV);
print("Username: $username Slurm Partition: $partition Command: $command Count: $count\n");
SingleRun::setPartition("$partition");
SingleRun::setUser("$username");

# if($#ARGV == 1){
#     SingleRun::greedyThrottleCommand($command, shift(@ARGV));
# }
# else {
#     SingleRun::throttleCommand($command);
# }

