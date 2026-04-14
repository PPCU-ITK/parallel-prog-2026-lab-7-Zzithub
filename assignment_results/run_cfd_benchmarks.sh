#!/usr/bin/env bash
set -euo pipefail

steps="${1:-2000}"
out_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${out_dir}/.." && pwd)"
csv="${out_dir}/cfd_euler_runtime_results.csv"

printf 'backend,Nx,Ny,nSteps,runtime_seconds,total_kinetic\n' > "${csv}"

for scale in 1 4 8 16; do
    nx=$((200 * scale))
    ny=$((100 * scale))
    for backend in cpu gpu; do
        exe="${repo_dir}/cfd_euler_${backend}"
        tmp="${out_dir}/raw_${backend}_${nx}x${ny}.txt"
        echo "Running ${backend} Nx=${nx} Ny=${ny} steps=${steps}" | tee "${tmp}"
        "${exe}" --backend "${backend}" --nx "${nx}" --ny "${ny}" --steps "${steps}" | tee -a "${tmp}"
        tail -n 1 "${tmp}" >> "${csv}"
    done
done

echo "Wrote ${csv}"
