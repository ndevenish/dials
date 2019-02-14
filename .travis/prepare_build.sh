#!/bin/bash

############################################################################
# Convenience definitions and functions
############################################################################

set -e

BOLD=$(tput bold)
NC=$(tput sgr0)
GREEN=$(tput setaf 2)
echot() {
    echo "${BOLD}${GREEN}$@${NC}"
}


############################################################################
# Install cmake generators and support repositories
############################################################################

cd ${TRAVIS_BUILD_DIR}

# Grab the autobuild repository
echot "Cloning custom cmake modules:"
git clone https://github.com/ndevenish/tbxcmake.git cmake
echot "Installing tbx conversion tools"
pip install -q --user git+https://github.com/ndevenish/tbxtools.git


############################################################################
# Get the rest of cctbx
############################################################################

echot "Grabbing remaining distribution modules:"
# Get all the other modules we need to compile
python cmake/prepare_singlemodule.py --write-log --no-cmake


############################################################################
# Generate/refresh the build
############################################################################

echot "Generating CMakeLists"
tbx2cmake . .
cp cmake/CMakeLists.txt CMakeLists.txt

echot "Updating timestamps"

# Handle timestamp updating from the cache. Since travis gives a fresh checkout
# every time, all files in the repository will be newer than the build cache dates,
# and so everything will be rebuilt anyway. This backdates things that haven't changed.
if [[ -f build/commit_ids.txt ]]; then
    # Use the old module list; any new modules won't need to be touched anyway
    MODULES=$(cat build/commit_ids.txt | awk '{ print $1; }')
    # Find the oldest time in the build directory
    echo "Oldest time: "
    find -type f -printf '%T+ %p\n' | sort | head -n 1

    OLDEST_MTIME=$(find . -type f -printf "%.10T@\n" | sort | head -n 1)
    echo "Backdating to $(date -d @$OLDEST_MTIME)"
    OLDEST_TS=$(date -d @$OLDEST_MTIME +%Y%m%d%H%M.%S)

    # Change the mtime of ALL checked out files to match this
    for repo in $MODULES; do
        echo "  Handling cache backdate for $repo"
        # Find the old commit and if it exists in the repository, we can backdate
        OLDREV=$(cat commit_ids.txt | grep "${repo} " | awk '{ print $2; }')
        if GIT_DIR=$repo/.git git cat-file -e $OLDREV; then
            # The commit exists. Backdate everything...
            find $repo | xargs touch -t $OLDEST_TS
            # And then forward-date everything that has changed
            GIT_DIR=$repo/.git git diff --name-only | xargs touch -c
        else
            echo "  Commit $OLDREV does not exist in repository; not backdating"
        fi
    done
else
    echo "No build/commit_ids.txt; Not syncing timestamps"
fi
# Finally, replace the old commit id file
mkdir -p build
mv commit_ids.txt build/

echo "CMake Options: ${CMAKE_OPTIONS}"

# Temporary debugging in case CMake finds something odd?
echo "All python versions in path:"
python -c 'import os; print([[path+"/"+x for x in os.listdir(path) if os.path.isfile(os.path.join(path,x)) and x.startswith("python")] for path in os.environ["PATH"].split(":") if os.path.isdir(path)])'

# Will do configure in separate travis line entry