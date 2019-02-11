#!/bin/bash

set -e

BOLD=$(tput bold)
NC=$(tput sgr0)
GREEN=$(tput setaf 2)
echot() {
    echo "${BOLD}${GREEN}$@${NC}"
}

cd ${TRAVIS_BUILD_DIR}

echo "Working build directory:"
ls
# Grab the autobuild repository
echot "Cloning custom cmake modules$"
git clone https://github.com/ndevenish/tbxcmake.git cmake
echot "Installing tbx conversion tools"
pip install --user git+https://github.com/ndevenish/tbxtools.git
echot "Grabbing remaining distribution modules:"
# Get all the other modules we need to compile
python cmake/prepare_singlemodule.py --write-log --no-cmake

echot "Generating CMakeLists"
tbx2cmake . .
cp cmake/CMakeLists.txt CMakeLists.txt

# IF we have no cache, then we want to use sccache to build - this means that
# we can survive over extra terminations (on OSX)
# - |
#   if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
#     if [[ ! -f build/build_complete ]]; then
#       echo "FIRST CACHE BUILD: USING S3"
#       export USE_SCCACHE=1
#       export CXX="sccache c++"
#       export CC="sccache cc"

#       export SCCACHE_BUCKET=ndtraviscache
#       sccache --start-server
#       echo "Started SCCACHE server"
#       sccache -s
#     else
#       echo "build_complete marker exists - not using s3-cache"
#     fi
#   fi

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
    if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
    echo "Backdating to $(date -d @$OLDEST_MTIME)"
    OLDEST_TS=$(date -d @$OLDEST_MTIME +%Y%m%d%H%M.%S)
    else
    echo "Backdating to $(date -r $OLDEST_MTIME)"
    OLDEST_TS=$(date -r $OLDEST_MTIME +%Y%m%d%H%M.%S)
    fi

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

# Do the actual building
mkdir -p build
# cd build

# Always give coloured output with CMake here
export CLICOLOR_FORCE=1
echo "CMake Options: ${CMAKE_OPTIONS}"

# Will do configure in separate travis line entry



# Show estimates of elapsed time whilst running
# - (while true; do python -c 'import os, time; t=time.time()-float(os.environ["START_TIME"]); print("\nEstimated Elapsed {:2.0f}:{:02.0f}s".format(t//60, t-(t//60)*60))'; sleep 20; done)&
# - export TIMER_PID=$!

# - |
#   if [[ -n $USE_SCCACHE ]]; then
#     sccache -s
#     sccache --stop-server
#   fi
