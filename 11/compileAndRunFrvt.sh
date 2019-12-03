echo "[MAKE] ./config folder.."
mkdir config
echo "[MAKE] ./detectFail folder.."
mkdir detectFail
echo "[MAKE] ./FAR folder.."
mkdir FAR
echo "[MAKE] ./FRR folder.."
mkdir FRR
echo "[DELETE] ./build files.."
rm -rf ./build/*
echo "[DELETE] ./src/nullImpl/build files.."
rm -rf ./src/nullImpl/build/*
echo "[DELETE] ./detectFail files.."
rm -rf ./detectFail/*
echo "[DELETE] ./FAR files.."
rm -rf ./FAR/*
echo "[DELETE] ./FRR files.."
rm -rf ./FRR/*
echo "[DELETE] ./lib files.."
rm -rf ./lib/*
echo "[DELETE] ./validation files.."
rm -rf ./validation/*
echo "[EXECUTE] ./scripts/build_null_impl.sh.."
./scripts/build_null_impl.sh
echo "[EXECUTE] ./run_validate_11.sh.."
./run_validate_11.sh
