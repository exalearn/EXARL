#!/usr/bin/perl
package SingleRun;
use strict;
use Time::HiRes qw(usleep);

my $user = "suet688";
my $partition = "newell";
my $overSubFactor = 2;
my $defaultThreashold = 1;
my $threashold = $defaultThreashold;

sub setPartition
{
    $partition = $_[0];
}

sub setUser
{
    $user = $_[0];
}

sub runCommand
{
    system($_[0]);
}

sub getCurrentAvailableNodes
{
    my $command = "sinfo -p $partition -t idle -o %D -h";
    my $output = `$command`;
    chomp $output;
    return $output
}

sub getCurrentTotalNodes
{
    my $command = "sinfo -p $partition -t idle,alloc -o %D -h";
    my $output = `$command`;
    chomp $output;
    return $output
}

sub getCurrentLoad
{
    my $command = "squeue -h -p $partition -u $user | wc -l";
    my $output = `$command`;
    chomp $output;
    return $output
}

sub getCurrentRunning
{
    my $command = "squeue -h -p $partition -u $user -t R | wc -l";
    my $output = `$command`;
    chomp $output;
    return $output
}

sub throttleCommand
{   
    my $command = $_[0];
    my $total = getCurrentTotalNodes();
    my $avail = getCurrentAvailableNodes();
    my $load = getCurrentLoad();
    if($threashold < $avail)
    {
        $threashold = int($avail*$overSubFactor);
    }
    else
    {
        my $max = $total * $overSubFactor;
        if($max < $threashold)
        {
            $threashold = $max;
        }
    }
    my $secs = 100;
    while($load >= $threashold)
    {
        usleep($secs);
        if($secs < 1000000)
        {
            $secs+=100;
        }
        $load = getCurrentLoad();
    }
    my $load = getCurrentLoad();
    runCommand($command);
    $total = getCurrentTotalNodes();
    $avail = getCurrentAvailableNodes();
    $load = getCurrentLoad();
    my $running = getCurrentRunning();
    my $percentage = 100 * $running / $total;
    print("Nodes: $total Avail: $avail Load: $load Percent: $percentage\n");
}

sub waitUntilFileExists
{
    my $fileName = shift(@_);
    my $flag = 0;
    while(!$flag)
    {
        if(-e $fileName)
        {
            $flag = 1;
        }
        else
        {
            # print("Waiting for $fileName\n");
            usleep(100);
        }
    }
}

sub waitUntilFileExistsTimeout
{
    my $fileName = shift(@_);
    my $waitTime = shift(@_) * 1000000;
    my $flag = 0;
    while(!$flag && $waitTime!=0)
    {
        if(-e $fileName)
        {
            $flag = 1;
        }
        else
        {
            usleep(100);
            $waitTime -= 100;
        }
    }
}
            
sub waitUntilDirExists
{
    my $dirName = shift(@_);
    my $flag = 0;
    while(!$flag)
    {
        if(-d $dirName)
        {
            $flag = 1;
        }
        else
        {
            # print("Waiting for $dirName\n");
            usleep(100);
        }
    }
}

sub getDirsWithPrefix
{
    my $root = shift(@_);
    my $prefix = shift(@_);

    opendir my $dh, $root 
        or die "$0: opendir: $!";
    my @dirs = grep {-d "$root/$_" && ! /^\.{1,2}$/} readdir($dh);
    return grep(/$prefix/, @dirs);
}

sub countDirsWithPrefix
{
    return scalar(getDirsWithPrefix(shift(@_)));
}

1;