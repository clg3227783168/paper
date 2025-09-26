export SOURCECODE_PATH=~/data/opentenbase/OpenTenBase
export INSTALL_PATH=~/data/opentenbase/install/
cd ~/data/opentenbase/
cd ${SOURCECODE_PATH}
rm -rf ${INSTALL_PATH}/opentenbase_bin_v5.0
chmod +x configure*
./configure --prefix=${INSTALL_PATH}/opentenbase_bin_v5.0 --enable-user-switch --with-libxml --disable-license --with-openssl --with-ossp-uuid CFLAGS="-g"
make clean
make -j1
make install
chmod +x contrib/pgxc_ctl/make_signature
cd contrib
make -j4
make install