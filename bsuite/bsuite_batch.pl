#!/usr/bin/perl
use strict;
use Cwd qw(cwd);
use File::Basename;
use FindBin;
use lib "$FindBin::RealBin";
require SingleRun;
use Getopt::Std;
my %options=();
my $dir = cwd;
my $script_path = dirname($0);

# ------------------ User options section ------------------ #
sub usage() {
    print "usage: bsuite_batch.pl [options] partition\n";
    print " partition: Slurm parition to submit jobs to\n\n";
    print " options:\n";
    print "  Slurm options:\n";
    print "    -N: Number of nodes per srun job\n";
    print "    -n: Number of ranks per srun job\n";
    print "    -a: Additional slurm options to pass\n";
    print "        (i.e. \"-x node15,node28\")\n";
    print "    -S: Launch jobs using sbatch template\n";
    print "  Exarl options:\n";
    print "    -P: Path to exarl root\n";
    print "    -s: Number of seeds per experiment\n";
    print "    -e: Number of episodes per experiment\n";
    print "    -p: Number of steps per episode per experiment\n";
    print "    -o: Output dir\n";
    print "    -A: Additional Exarl options\n";
    print "        (i.e. \"--agent async\")\n";
    print "  Bsuite options:\n";
    print "    -b: Bsuite subset to run (defualt is all)\n";
    print "    -B: Only print bsuite subset, seeds, and episodes\n";
    print "        To see all use subset all (i.e. -B all)\n";
    print "    -D: Display bsuite subsets.\n";
    print "  Script options:\n";
    print "    -t: Turn OFF throttling\n";
    print "    -x: Don't run just print jobs\n";
    print "    -c: Create directories but do not run\n";
    print "    -h: Print this message\n\n";
    print "Example usage:\n";
    print "  ./bsuite/bsuite_batch.pl -N 2 -n 2 -a \"-x node15,node28,node22,node42,node33\" -b developer -o out -s 2 -e 100 -p 100 slurm\n";
    print "  ./bsuite/bsuite_batch.pl -N 2 -n 2 -A \"--agent async\" -o out slurm\n";
    print "  ./bsuite/bsuite_batch.pl -N 1 -n 1 -S ./script/cori_V100_gpu.sh -b memory -s 1 -o out slurm\n";
    print "  ./bsuite/bsuite_batch.pl -D slurm\n";
    print "  ./bsuite/bsuite_batch.pl -B all slurm\n";
    exit()
}

# User defined options
getopts("N:n:a:P:s:e:p:u:p:o:A:b:B:S:txhcD",\%options) or usage;
my $N = defined $options{N} ? $options{N} : 1;
my $n = defined $options{n} ? $options{n} : 1;
my $a = defined $options{a} ? $options{a} : "";
my $path = defined $options{P} ? $options{P} : ".";
my $seeds = defined $options{s} ? $options{s} : 1000000000;
my $episodes = defined $options{e} ? $options{e} : 1000000000;
my $steps = defined $options{p} ? "--n_steps $options{p}" : "";
my $A = defined $options{A} ? $options{A} : "";
my $throttle = defined $options{t} ? 0 : 1;
my $run = defined $options{x} ? 0 : 1;
my $bsuite_set = defined $options{b} ? $options{b} : "";
my $output_dir = defined $options{o} ? $options{o} : ".";
my $make_dir = defined $options{c} ? $options{c} : 0;
if(defined $options{h}) {
    usage();
}
if(defined $options{B}) {
    $bsuite_set = $options{B};
    my $path_to_py_module = $script_path . "/bsuite_all.py " . $bsuite_set;
    my $bsuite_txt = `python $path_to_py_module`;
    print("Subset: $bsuite_set\n");
    print("Name Seeds Episodes\n");
    print($bsuite_txt);
    exit();
}
if(defined $options{D}) {
    my $path_to_py_module = $script_path . "/bsuite_all.py display";
    my $bsuite_txt = `python $path_to_py_module`;
    print("Subsets:\n");
    print($bsuite_txt);
    exit();
}
my @template;
if(defined $options{S}) {
    my $file = $options{S};
    open(my $fh, "<", $file) or die "could not open $file: $!";
    chomp(@template = <$fh>);
    close($fh);
    
    for(my $i=0; $i<=$#template; $i++) {
        if($template[$i] =~ /#SBATCH/) {
            if($template[$i] =~ /-n/) {
                $template[$i] = "#SBATCH -n $n"
            }
        }
    }
}

# ------------------ Function section ------------------ #

# Wrapper to make directories
sub makeDir {
    if($run || $make_dir) {
        my $path_to_make = shift(@_);
        if(-d $path_to_make){
            print("Dir $path_to_make exists.\n");
        }
        else {
            print("Making $path_to_make\n");
            system("mkdir $path_to_make");
        }
    }
}

# This functions calls a python module to get the benchmark, seed, episodes
# that are found in the bsuite python module.  script_path is the dir that
# this perl script is found in.  Make sure bsuite_all and this script are
# colocated!
sub getBsuiteBenchSet {
    my $path_to_py_module = $script_path . "/bsuite_all.py " . $bsuite_set;
    my $ret = `python $path_to_py_module`;
    my @lines = split("\n", $ret);
    my %bsuite_benchmarks;
    foreach my $line (@lines) {
        my @parts = split(" ", $line);
        $bsuite_benchmarks{$parts[0]} = [$parts[1], $parts[2]];
    }
    return %bsuite_benchmarks
}

# This function checks to see if there is a directory called
# bench_seed_episode under the provided path.  If there is not
# then it creates the directory and returns the path.
sub checkForResults {
    my $bench = shift(@_);
    my $seed = shift(@_);
    my $episode = shift(@_);
    my $out_dir = shift(@_);
    my $to_check = $out_dir . "/" . $bench . "_" . $seed . "_" . $episode; 
    if(-d $to_check) {
        return 0;
    }
    if($run) {
        system("mkdir $to_check");
    }
    return $to_check;
}

# This function generates a slurm command if results do not already exist.
sub getSbatchCommand {
    my $bench = shift(@_);
    my $seed = shift(@_);
    my $episode = shift(@_);
    my $partition = shift(@_);
    my $driver_path = $path . "/exarl/driver";

    my $bench_dir_name = $bench;
    my $outfile = $bench_dir_name . "/" . $bench .  "_" . $seed . "_" . $episode . ".txt";
    
    if(defined $options{o}) {
        $bench_dir_name = $output_dir . "/" . $bench_dir_name;
        $outfile = $output_dir . "/" . $outfile;
    }

    makeDir($bench_dir_name);
    my $exp_dir = checkForResults($bench, $seed, $episode, $bench_dir_name);
    if($exp_dir) {
        makeDir($exp_dir);
        my $output = "--output_dir $bench_dir_name";

        # Set the srun command in script
        for(my $i=0; $i<=$#template; $i++) {
            if($template[$i] =~ /srun/) {
                $template[$i] = "srun python $driver_path --env Bsuite-v0 --bsuite_id $bench --seed_number $seed --n_episodes $episode $steps $A $output&> $outfile";
            }
        }

        # Write script
        my $filename = $bench_dir_name . "/" . $bench .  "_" . $seed . "_" . $episode . ".sh";
        my $to_write = join("\n", @template);
        print("$filename \n");
        open(my $fh, '>', $filename) or die $!;
        print $fh $to_write;
        close($fh);

        return "sbatch -N $N $filename"; 
    }
    return 0;
}

# This function generates a slurm command if results do not already exist.
sub getSlurmCommand {
    my $bench = shift(@_);
    my $seed = shift(@_);
    my $episode = shift(@_);
    my $partition = shift(@_);
    my $driver_path = $path . "/exarl/driver";

    my $bench_dir_name = $bench;
    my $outfile = $bench_dir_name . "/" . $bench .  "_" . $seed . "_" . $episode . ".txt";
    
    if(defined $options{o}) {
        $bench_dir_name = $output_dir . "/" . $bench_dir_name;
        $outfile = $output_dir . "/" . $outfile;
    }

    makeDir($bench_dir_name);
    my $exp_dir = checkForResults($bench, $seed, $episode, $bench_dir_name);
    if($exp_dir) {
        makeDir($exp_dir);
        my $output = "--output_dir $bench_dir_name";

        return "srun -p $partition -N $N -n $n $a python $driver_path --env Bsuite-v0 --bsuite_id $bench --seed_number $seed --n_episodes $episode $steps $A $output&> $outfile &";
    }
    return 0;
}

sub getCommand {
    if(@template) {
        return getSbatchCommand(@_);
    }
    return getSlurmCommand(@_);
}

# ------------------ Scripting section ------------------ #

# These are used to submit and throttle commands sent to slurm
my $username = $ENV{LOGNAME} || $ENV{USER} || getpwuid($<);
my $partition  = shift(@ARGV);
print("Username: $username Slurm Partition: $partition\n");
SingleRun::setPartition("$partition");
SingleRun::setUser("$username");

# Create output directory
makeDir($output_dir);

# This hashmap of desired benchmarks.
# Benchmark => (Number of seeds, Number of Episodes)
my %bsuite_bench = getBsuiteBenchSet();

# Iterate over all benchmarks
foreach my $benchmark (keys %bsuite_bench) {
    # Get the min seed and episode
    my $min_seed = $bsuite_bench{$benchmark}[0] <= $seeds ? $bsuite_bench{$benchmark}[0] : $seeds;
    my $min_episode = $bsuite_bench{$benchmark}[1] <= $episodes ? $bsuite_bench{$benchmark}[1] : $episodes;
    for(my $i=0; $i<$min_seed; $i++) {
        my $command = getCommand($benchmark, $i, $min_episode, $partition);
        if($command) {
            if($run) {
                # If we don't use throttling, all jobs will be dumped into the system
                if($throttle) {
                    SingleRun::throttleCommand($command);
                }
                else {
                    print("$command\n");
                    SingleRun::runCommand($command);
                }
            }
            else {
                print("$command\n");
            }
        }
        else {
            print("Skipping $benchmark $i $min_episode\n");
        }
    }
}
