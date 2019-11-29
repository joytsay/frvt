echo "[MAKE] ./config folder.."
mkdir config
echo "[MAKE] ./detectFail folder.."
mkdir detectFail
echo "[DELETE] ./build files.."
rm -rf ./build/*
echo "[DELETE] ./src/nullImpl/build files.."
rm -rf ./src/nullImpl/build/*
echo "[DELETE] ./lib files.."
rm -rf ./lib/*
echo "[EXECUTE] ./scripts/build_null_impl.sh.."
./scripts/build_null_impl.sh
echo "[EXECUTE] ./run_validate_11.sh.."
./run_validate_11.sh
