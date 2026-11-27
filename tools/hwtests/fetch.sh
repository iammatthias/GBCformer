#!/bin/sh
# Fetches the outside test ROMs into vendor/hwtests/ (gitignored), each pinned by a
# release checksum or a commit, and checks every download before unpacking it:
#   gbtr/     c-sp/game-boy-test-roms v7.0: mooneye-test-suite, SameSuite, Mealybug
#             Tearoom, blargg, cgb-acid-hell and others, prebuilt, with each suite's
#             license and the screenshots the suites judge by.
#   daid/     testroms/daid of gbdev/GBEmulatorShootout (MIT) at a fixed commit.
#   sameboy/  SameBoy v1.0.5's own boot ROMs (MIT, written from scratch), the open
#             substitute `--boot` is tested with. Nintendo's boot ROMs are never fetched.
# Idempotent: a directory already holding its stamp is left alone.
set -eu
cd "$(dirname "$0")/../.."
root=vendor/hwtests
mkdir -p "$root"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fetch() { # url sha256 out
  curl -sSfL -o "$3" "$1"
  got=$(shasum -a 256 "$3" | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "$1: sha256 $got, want $2" >&2; exit 1; }
}

if [ ! -f "$root/gbtr/.stamp" ]; then
  fetch https://github.com/c-sp/game-boy-test-roms/releases/download/v7.0/game-boy-test-roms-v7.0.zip \
    b9a9d7a1075aa35a3d07c07c34974048672d8520dca9e07a50178f5860c3832c "$tmp/gbtr.zip"
  rm -rf "$root/gbtr" && mkdir "$root/gbtr"
  unzip -q "$tmp/gbtr.zip" -d "$root/gbtr"
  echo "c-sp/game-boy-test-roms v7.0" > "$root/gbtr/.stamp"
fi

commit=38b926bdbc26993d1b4c43e97979ecc66287bf02
if [ ! -f "$root/daid/.stamp" ]; then
  rm -rf "$root/daid" && mkdir "$root/daid"
  while read -r f sum; do
    case $f in LICENSE) url=https://raw.githubusercontent.com/gbdev/GBEmulatorShootout/$commit/LICENSE ;; *) url=https://raw.githubusercontent.com/gbdev/GBEmulatorShootout/$commit/testroms/daid/$f ;; esac
    fetch "$url" "$sum" "$root/daid/$f"
  done <<EOF
stop_instr.gb 7807e380ce4531f4f9425427b47d7216a92ef2283ab706ea4a33eb84f589ac42
stop_instr.dmg.png 4a17cdbaa7d108eab02da64e95bb08b350f32b290819319af9d6c611b66bb1b0
stop_instr.gbc.png dde322b8d635522880d31e8a336c3f6d2308e0b943456bf13279faab89502614
stop_instr_gbc_mode3.gb a16550bc85a2f5f8462694094e49cfd087780d731a4148e9056adfa3fbda7cff
stop_instr_gbc_mode3.png 63bab5e13e22dadaa1257de56a02e5dc0a51f5cb7239cb83133d3daa5aeea37d
ppu_scanline_bgp.gb 3a32e875d4cdc1defd28a601a64a9a3f1f7965a6679a8f16af45781520b57379
ppu_scanline_bgp.gbc.png 49e0438aca7c762b8392db65ee77a59e55155925624461a161e9530a982fb25b
ppu_scanline_bgp_0.dmg.png 146595658062ef1e2fc19b46b665a997d0e6d44cab9bdade31232f5c997918e8
ppu_scanline_bgp_1.dmg.png a07067f06285f94180a246b04208223a01040014b750996de1bbed3eaf56b572
ppu_scanline_bgp_2.dmg.png 2400f19f5a82d1c28726cdbb9793f86f01cadb1bbc5ef624e37e69de272241cc
rom_and_ram.gb 915ee4e41db0a10e34d22af1acd9ff193bd688448eba597d53460dacfdec7ae4
speed_switch_timing_div.gbc 7569d79d9bf2918f8f5a930f01643358a87f58a5fb0dfd33ee82d5b4badc5e94
speed_switch_timing_div.png acb8bd2ccb3e0c720946d6c2365eb58116e627cceea74dd02c06408859253184
speed_switch_timing_ly.gbc acbe324c11b5e168d34c546cb9f3e641f3e85faffac3108fa0a933f6e6b95fb2
speed_switch_timing_ly.png a64252ec9e245773ffc8dc1dab3712fe3be11fd5a5081cc722dab61461165dc0
speed_switch_timing_stat.gbc 21e088c4d4c110ef76912a2c4fd05293ac8cbf22740b35f2e362ee7b5450c1be
speed_switch_timing_stat.png 9bc6fd0feef711611423f047ce8d3d0bceb9894ec96de6ac5b96938ad6691b3c
LICENSE 669d3638cdf61080ce95c612157b0643868f7be750fe8dcadc2e2685a0b4e994
EOF
  echo "gbdev/GBEmulatorShootout $commit testroms/daid" > "$root/daid/.stamp"
fi

if [ ! -f "$root/sameboy/.stamp" ]; then
  fetch https://github.com/LIJI32/SameBoy/releases/download/v1.0.5/sameboy_winsdl_v1.0.5.zip \
    5aa92fb376b53b57361b15c29a6ff5a4494f7ae6f4ba7cbdfa6ef8375765b655 "$tmp/sameboy.zip"
  rm -rf "$root/sameboy" && mkdir "$root/sameboy"
  unzip -q -j "$tmp/sameboy.zip" dmg_boot.bin cgb_boot.bin -d "$root/sameboy"
  fetch https://raw.githubusercontent.com/LIJI32/SameBoy/v1.0.5/LICENSE 875b2c940181c315fa99ce5292326b985a43cdc4e878b38995066d6b495ff151 "$root/sameboy/LICENSE"
  echo "LIJI32/SameBoy v1.0.5 boot ROMs" > "$root/sameboy/.stamp"
fi
echo "vendor/hwtests: $(find "$root" -name '*.gb*' | wc -l | tr -d ' ') ROMs"
