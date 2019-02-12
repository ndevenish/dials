#!/bin/bash

set -e

BOLD=$(tput bold)
NC=$(tput sgr0)
GREEN=$(tput setaf 2)
echot() {
    echo "${BOLD}${GREEN}$@${NC}"
}

###############################################################################
# before-install

# Rewrite the path on OSX so that we use the homebrew, rather than system, python
# Also: non-prefixed versions of gnu find utilities
# Define (possibly platform-specific) variables for build
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    # export PATH="/usr/local/opt/python/libexec/bin:$PATH"
    export PATH="/usr/local/opt/findutils/libexec/gnubin:$PATH"
    export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"
fi
export START_TIME=$(date +%s)

# Run a command, but stop it before we run out of travis time
travis_timeout() {

    # Periodically remind how long the job seems to have been running
    (
        while true; do
            python -c 'import os, time; t=time.time()-float(os.environ["START_TIME"]); print("\nEstimated Elapsed {:2.0f}:{:02.0f}s".format(t//60, t-(t//60)*60))'
            sleep 20
        done
    )&
    export TIMER_PID=$!
    # 2520: 42 minutes
    timeout -k 10 "$((2520-($(date +%s)-$START_TIME)))"  "$@"
    # Save the return value so that we can pass it through to after killing timer
    actual_ret=$?
    kill -9 $TIMER_PID || true
    return actual_ret
}

# Do a step, and show the command
step() {
    (
        set -x
        "$@"
    )
}

echot "Python versions:"
echo "python  $(python --version 2>&1  | awk '{ print $2; }') ($(which python))"
echo "python2 $(python2 --version 2>&1 | awk '{ print $2; }') ($(which python2))"
echo "python3 $(python3 --version 2>&1 | awk '{ print $2; }') ($(which python3))"

# # Update homebrew
# if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
#     echot "Updating homebrew:"
#     # HOMEBREW_NO_AUTO_UPDATE=1 brew info cmake eigen hdf5 || true
#     brew update > /dev/null;
#     echo "Installed/available packages:"
#     brew info --json cmake eigen hdf5 | \
#         python2 -c 'import sys, json; print("             Ver   Avail\n"+"\n".join(["{name:8}{linked_keg:>8}{versions[stable]:>8}".format(**{k:y if y is not None else "" for (k, y) in x.items()}) for x in json.load(sys.stdin)]))' || true
#     export HOMEBREW_NO_AUTO_UPDATE=1
# fi

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    # New more intelligent way to ensure brew dependencies
    # - if an update is not required, it will not do one
    # - it will only upgrade packages that it needs to
    # - currently doesn't handle unlinked kegs well
    step brew info --json cmake
    echo "Now you"
    python ${TRAVIS_BUILD_DIR}/.travis/resolve_brew_dependencies.py \
        'cmake>=3.12' 'eigen>=3.2.8,<4' coreutils findutils 'hdf5~=1.10'

fi
###############################################################################
# install

############################################################################
# All the dependencies are installed in ${TRAVIS_BUILD_DIR}/deps/
############################################################################
DEPS_DIR="${TRAVIS_BUILD_DIR}/deps"

mkdir -p ${DEPS_DIR}
# For some reason this sets off a bug in rubys magic cd override in the
# travis xcode10.1 image - and the cd fails (even though the path exists).
# Although, we no longer need to move to this path explicitly?
# && cd ${DEPS_DIR} || true

############################################################################
# Setup default versions and override compiler if needed
############################################################################
if [[ "${BOOST_VERSION}" == "default" ]]; then
    BOOST_VERSION=1.63.0;
fi
# # Update dependencies from homebrew
# if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then # Do OSX brew dependencies
#     echot "Upgrading homebrew packages:"
#     brew cask uninstall oclint || true # Fix bug where this overwrites poured links
#     for package in cmake eigen findutils hdf5 coreutils; do
#     if brew ls --versions $package > /dev/null; then
#         brew outdated $package || brew upgrade $package
#     else
#         brew install $package
#     fi
#     done
# fi

# Other python libs we know about - need numpy before boost is built
echot "Python libraries for build"
step pip install --user mock docopt pathlib2 enum34 pyyaml ninja numpy

############################################################################
# Build/Install specified boost version with boost-python
############################################################################

# Install requested boost version
if [[ "${BOOST_VERSION}" != "" ]]; then
    echot "Ensuring Boost-${BOOST_VERSION}:"
    BOOST_DIR=${DEPS_DIR}/boost-${BOOST_VERSION}
    BOOST_BUILD_DIR=~/build_tmp/boost
    if [[ -z "$(ls -A ${BOOST_DIR} 2>/dev/null)" ]]; then
    if [[ "${BOOST_VERSION}" == "trunk" ]]; then
        BOOST_URL="http://github.com/boostorg/boost.git"
        travis_retry git clone --depth 1 --recursive ${BOOST_URL} ${BOOST_BUILD_DIR} || exit 1
    else
        BOOST_URL="http://sourceforge.net/projects/boost/files/boost/${BOOST_VERSION}/boost_${BOOST_VERSION//\./_}.tar.gz"
        mkdir -p ${BOOST_BUILD_DIR}
        { travis_retry wget -nv -O - ${BOOST_URL} | tar --strip-components=1 -xz -C ${BOOST_BUILD_DIR}; } || exit 1
    fi
    mkdir -p ${BOOST_DIR}
    (cd ${BOOST_BUILD_DIR} && ./bootstrap.sh --with-python=$(which python2) && ./b2 -j 3 -d0 --prefix=${BOOST_DIR} --with-python --with-atomic --with-thread --with-chrono --with-date_time install) || exit 1
    else
    echo "Boost exists in ${BOOST_DIR}; using existing"
    fi
    CMAKE_OPTIONS+=" -DBOOST_ROOT=${BOOST_DIR}"
fi

############################################################################
# Install a recent CMake
############################################################################
# Install/upgrade cmake
if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
    CMAKE_MAJOR=3.10
    CMAKE_POINT_RELEASE=3.10.0
    echot "Ensuring CMake==${CMAKE_POINT_RELEASE}:"
    # If the path for cmake exists, check if it is the right version
    if [[ -n "$(ls -A ${DEPS_DIR}/cmake/bin 2>/dev/null)" ]]; then
    if [[ $(cmake/bin/cmake --version | head -n1 | awk '{ print $3; }') != "${CMAKE_POINT_RELEASE}" ]]; then
        echo "CMake out of date. Removing so we can recreate."
        rm -rf cmake
    fi
    fi
    # If the path doesn't exist, then download cmake
    if [[ -z "$(ls -A ${DEPS_DIR}/cmake/bin 2>/dev/null)" ]]; then
    CMAKE_URL="https://cmake.org/files/v${CMAKE_MAJOR}/cmake-${CMAKE_POINT_RELEASE}-Linux-x86_64.tar.gz"
    mkdir -p cmake && travis_retry wget --no-check-certificate --quiet -O - "${CMAKE_URL}" | tar --strip-components=1 -xz -C cmake
    fi
    export PATH="${DEPS_DIR}/cmake/bin:${PATH}"
else
    if ! brew ls --version cmake &>/dev/null; then brew install cmake; fi
fi

# step cmake --version | head -n1
echo "$(cmake --version | head -n1)"

############################################################################
# Build/Install specified HDF5 version
############################################################################
# - |
#   if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then # Install HDF5
#     HDF5_VERSION=1.10.1
#     HDF5_URL=https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-${HDF5_VERSION}.tar.bz2
#     HDF5_DIR=${DEPS_DIR}/hdf5-${HDF5_VERSION}
#     HDF5_BUILD_DIR=~/build_tmp/hdf5
#     if [[ ! -d ${HDF5_DIR} ]]; then
#       mkdir -p ${HDF5_BUILD_DIR} && travis_retry wget --quiet --no-check-certificate --quiet -O - "${HDF5_URL}" | tar --strip-components=1 -x -C ${HDF5_BUILD_DIR}
#       ( mkdir -p $HDF5_BUILD_DIR/_build
#         cd ${HDF5_BUILD_DIR}/_build
#         cmake .. -DCMAKE_INSTALL_PREFIX=${HDF5_DIR} -DBUILD_TESTING=off -DCMAKE_BUILD_TYPE=Release
#         cmake --build . --target install -- -j 3
#         )
#     fi
#     CMAKE_OPTIONS+=" -DHDF5_ROOT=${HDF5_DIR}"
#   fi

# On OSX, make sure that the --user pip is on the path
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then # Update PATH for pip --user
    export PATH=$(python -c "import site, os; print(os.path.join(site.USER_BASE, 'bin'))"):$PATH
fi

# Move the current repository into a dials subdirectory
echot "Moving repository to subdirectory dials/"
(
    set -x
    builtin cd ${TRAVIS_BUILD_DIR}
    mkdir dials && mv $(git ls-tree --name-only HEAD) dials && mv .git dials/
)

set +e