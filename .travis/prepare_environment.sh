#!/bin/bash

############################################################################
# Configuration paths and variables
############################################################################

# Where all the custom dependencies are built into
DEPS_DIR="${TRAVIS_BUILD_DIR}/deps"
# If no Boost specified, install this version
BOOST_DEFAULT_VERSION=1.63.0
# Specifier for the version of CMake to use
CMAKE_SPEC='>=3.12'
# The version of msgpack to prefer
MSGPACK_VERSION=3.1.1


############################################################################
# Convenience definitions and functions
############################################################################

set -e

# First thing we do is record the start time
export START_TIME=$(date +%s)

# Coloured output
BOLD=$(tput bold)
NC=$(tput sgr0)
GREEN=$(tput setaf 2)
echot() {
    echo "${BOLD}${GREEN}$@${NC}"
}

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
    return $actual_ret
}

# Do a step, and show the command
step() {
    (
        set -x
        "$@"
    )
}

# Validate a version number against a PEP508-style specifier
validate_spec() {
    version=$1
    shift
    spec=$*
    python - <<DOC
import sys
try:
    from packaging.version import parse
    from packaging.specifiers import SpecifierSet
except ImportError:
    from pip._vendor.packaging.version import parse
    from pip._vendor.packaging.specifiers import SpecifierSet

if parse("${version}") in SpecifierSet("""${spec}"""):
    sys.exit(0)
sys.exit(1)
DOC
}

############################################################################
# Set up paths and diagnostic information
############################################################################

# Use non-prefixed versions of gnu find utilities and coreutils
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    export PATH="/usr/local/opt/findutils/libexec/gnubin:$PATH"
    export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"
fi

mkdir -p ${DEPS_DIR}

echot "Python versions:"
echo "python  $(python --version 2>&1  | awk '{ print $2; }') ($(which python))"
echo "python2 $(python2 --version 2>&1 | awk '{ print $2; }') ($(which python2))"
echo "python3 $(python3 --version 2>&1 | awk '{ print $2; }') ($(which python3))"

# Don't try to install packages whilst building
export NEVER_INSTALL_REQUIRE=1

############################################################################
# Handle mac OSX dependencies via homebrew
############################################################################

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    # New more intelligent way to ensure brew dependencies
    # - if an update is not required, it will not do one
    # - it will only upgrade packages that it needs to
    # - currently doesn't handle unlinked kegs well
    python ${TRAVIS_BUILD_DIR}/.travis/resolve_brew_dependencies.py \
        "cmake${CMAKE_SPEC}" 'eigen>=3.2.8,<4' coreutils findutils 'hdf5~=1.10' \
        msgpack
    # Don't do numpy through homebrew - seems broken(?) on default image and
    # upgrading causes a whole painful chain of dependencies. postgis/gdal use
    # this but we don't care about them
    HOMEBREW_NO_AUTO_UPDATE=1 brew uninstall --ignore-dependencies numpy
fi

############################################################################
# Python paths and dependencies
############################################################################

# On OSX, make sure that the --user pip is on the path
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then # Update PATH for pip --user
    export PATH=$(python -c "import site, os; print(os.path.join(site.USER_BASE, 'bin'))"):$PATH
fi

# Other python libs we know about - need numpy before boost is built
echot "Python libraries for build"
step pip install -q --user mock docopt pathlib2 enum34 pyyaml ninja msgpack numpy
# Dependencies from libtbx.show_python_dependencies
step pip install -q --user \
    blosc Jinja2 'mock>=2.0' msgpack orderedset procrunner 'pytest>=3.1,>=3.6' \
    scikit_learn[alldeps] scipy tabulate 'tqdm==4.23.4' pytest-xdist

############################################################################
# Build/Install specified boost version with boost-python
############################################################################

if [[ "${BOOST_VERSION}" != "" ]]; then
    if [[ "${BOOST_VERSION}" == "default" ]]; then
        BOOST_VERSION=${BOOST_DEFAULT_VERSION}
    fi
    BOOST_DIR=${DEPS_DIR}/boost-${BOOST_VERSION}
    BOOST_BUILD_DIR=~/build_tmp/boost
    if [[ -z "$(ls -A ${BOOST_DIR} 2>/dev/null)" ]]; then
        echot "Installing Boost ${BOOST_VERSION}"
        if [[ "${BOOST_VERSION}" == "trunk" ]]; then
            BOOST_URL="http://github.com/boostorg/boost.git"
            travis_retry git clone --depth 1 --recursive ${BOOST_URL} ${BOOST_BUILD_DIR} || exit 1
        else
            BOOST_URL="http://sourceforge.net/projects/boost/files/boost/${BOOST_VERSION}/boost_${BOOST_VERSION//\./_}.tar.gz"
            mkdir -p ${BOOST_BUILD_DIR}
            { travis_retry wget -nv -O - ${BOOST_URL} | tar --strip-components=1 -xz -C ${BOOST_BUILD_DIR}; } || exit 2
        fi
        mkdir -p ${BOOST_DIR}
        (
            builtin cd ${BOOST_BUILD_DIR}
            ./bootstrap.sh --with-python=$(which python2)
            ./b2 -j 3 -d0 --prefix=${BOOST_DIR} --with-python --with-atomic --with-thread --with-chrono --with-date_time install
        ) || exit 3
    else
        echot "Using Boost-${BOOST_VERSION} in ${BOOST_DIR}"
    fi
    CMAKE_OPTIONS+=" -DBOOST_ROOT=${BOOST_DIR}"
fi

############################################################################
# Build/Install msgpack
############################################################################

if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
    MSGPACK_DIR=${DEPS_DIR}/msgpack-${MSGPACK_VERSION}
    MSGPACK_URL=https://github.com/msgpack/msgpack-c/releases/download/cpp-${MSGPACK_VERSION}/msgpack-${MSGPACK_VERSION}.tar.gz
    if [[ ! -d ${MSGPACK_DIR} ]]; then
        MSGPACK_BUILD_DIR=~/build_tmp/msgpack
        mkdir -p ${MSGPACK_BUILD_DIR}
        { travis_retry wget -nv -O - ${MSGPACK_URL} | tar --strip-components=1 -xz -C ${MSGPACK_BUILD_DIR}; } || exit 4
        (
            mkdir -p ${MSGPACK_BUILD_DIR}/_build
            builtin cd ${MSGPACK_BUILD_DIR}/_build
            cmake .. -DCMAKE_INSTALL_PREFIX=${MSGPACK_DIR} -DMSGPACK_BUILD_EXAMPLES=no
            make install
        ) || exit 5
    fi
    # CMAKE_OPTIONS+=" -Dmsgpack_ROOT=${MSGPACK_DIR}"
    # Prefer environment variable for now
    export msgpack_ROOT=${MSGPACK_DIR}
fi

############################################################################
# Install a recent CMake
############################################################################
# Install/upgrade cmake
if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
    step pip install -q --user cmake
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

# Move the current repository into a dials subdirectory
echot "Moving repository to subdirectory dials/"
(
    set -x
    builtin cd ${TRAVIS_BUILD_DIR}
    mkdir dials && mv $(git ls-tree --name-only HEAD) dials && mv .git dials/
)

# Always give coloured output with CMake
export CLICOLOR_FORCE=1

set +e