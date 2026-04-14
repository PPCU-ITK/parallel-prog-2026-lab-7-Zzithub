#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

enum class Backend {
    Cpu,
    Gpu
};

struct Config {
    int Nx = 200;
    int Ny = 100;
    int nSteps = 2000;
    Backend backend = Backend::Cpu;
    bool progress = false;
};

struct RunResult {
    double seconds = 0.0;
    double total_kinetic = 0.0;
};

#pragma omp declare target
static inline int idx(int i, int j, int stride) {
    return i * stride + j;
}

static inline double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

static inline void fluxX(double rho, double rhou, double rhov, double E,
                         double *frho, double *frhou, double *frhov, double *fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    *frho = rhou;
    *frhou = rhou * u + p;
    *frhov = rhov * u;
    *fE = (E + p) * u;
}

static inline void fluxY(double rho, double rhou, double rhov, double E,
                         double *frho, double *frhou, double *frhov, double *fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    *frho = rhov;
    *frhou = rhou * v;
    *frhov = rhov * v + p;
    *fE = (E + p) * v;
}
#pragma omp end declare target

static void usage(const char *prog) {
    cerr << "Usage: " << prog << " [--backend cpu|gpu] [--nx N] [--ny N] [--steps N] [--progress]\n";
}

static bool parse_args(int argc, char **argv, Config *cfg) {
    for (int a = 1; a < argc; ++a) {
        if (strcmp(argv[a], "--backend") == 0 && a + 1 < argc) {
            string value = argv[++a];
            if (value == "cpu") {
                cfg->backend = Backend::Cpu;
            } else if (value == "gpu") {
                cfg->backend = Backend::Gpu;
            } else {
                usage(argv[0]);
                return false;
            }
        } else if (strcmp(argv[a], "--nx") == 0 && a + 1 < argc) {
            cfg->Nx = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--ny") == 0 && a + 1 < argc) {
            cfg->Ny = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--steps") == 0 && a + 1 < argc) {
            cfg->nSteps = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--progress") == 0) {
            cfg->progress = true;
        } else if (strcmp(argv[a], "--help") == 0 || strcmp(argv[a], "-h") == 0) {
            usage(argv[0]);
            return false;
        } else {
            usage(argv[0]);
            return false;
        }
    }

    if (cfg->Nx <= 0 || cfg->Ny <= 0 || cfg->nSteps <= 0) {
        usage(argv[0]);
        return false;
    }
    return true;
}

static void apply_cpu_boundary_conditions(int Nx, int Ny, double rho0, double u0, double v0, double E0,
                                          double *rho, double *rhou, double *rhov, double *E) {
    const int stride = Ny + 2;

    #pragma omp parallel for
    for (int j = 0; j < Ny + 2; j++) {
        rho[idx(0, j, stride)] = rho0;
        rhou[idx(0, j, stride)] = rho0 * u0;
        rhov[idx(0, j, stride)] = rho0 * v0;
        E[idx(0, j, stride)] = E0;

        rho[idx(Nx + 1, j, stride)] = rho[idx(Nx, j, stride)];
        rhou[idx(Nx + 1, j, stride)] = rhou[idx(Nx, j, stride)];
        rhov[idx(Nx + 1, j, stride)] = rhov[idx(Nx, j, stride)];
        E[idx(Nx + 1, j, stride)] = E[idx(Nx, j, stride)];
    }

    #pragma omp parallel for
    for (int i = 0; i < Nx + 2; i++) {
        rho[idx(i, 0, stride)] = rho[idx(i, 1, stride)];
        rhou[idx(i, 0, stride)] = rhou[idx(i, 1, stride)];
        rhov[idx(i, 0, stride)] = -rhov[idx(i, 1, stride)];
        E[idx(i, 0, stride)] = E[idx(i, 1, stride)];

        rho[idx(i, Ny + 1, stride)] = rho[idx(i, Ny, stride)];
        rhou[idx(i, Ny + 1, stride)] = rhou[idx(i, Ny, stride)];
        rhov[idx(i, Ny + 1, stride)] = -rhov[idx(i, Ny, stride)];
        E[idx(i, Ny + 1, stride)] = E[idx(i, Ny, stride)];
    }
}

static double update_cpu(int Nx, int Ny, double dt, double dx, double dy, const bool *solid,
                         double *rho, double *rhou, double *rhov, double *E,
                         double *rho_new, double *rhou_new, double *rhov_new, double *E_new) {
    const int stride = Ny + 2;
    const double dtdx = dt / (2.0 * dx);
    const double dtdy = dt / (2.0 * dy);

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            const int c = idx(i, j, stride);
            if (solid[c]) {
                rho_new[c] = rho[c];
                rhou_new[c] = rhou[c];
                rhov_new[c] = rhov[c];
                E_new[c] = E[c];
                continue;
            }

            const int ip = idx(i + 1, j, stride);
            const int im = idx(i - 1, j, stride);
            const int jp = idx(i, j + 1, stride);
            const int jm = idx(i, j - 1, stride);

            rho_new[c] = 0.25 * (rho[ip] + rho[im] + rho[jp] + rho[jm]);
            rhou_new[c] = 0.25 * (rhou[ip] + rhou[im] + rhou[jp] + rhou[jm]);
            rhov_new[c] = 0.25 * (rhov[ip] + rhov[im] + rhov[jp] + rhov[jm]);
            E_new[c] = 0.25 * (E[ip] + E[im] + E[jp] + E[jm]);

            double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
            double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
            double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
            double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

            fluxX(rho[ip], rhou[ip], rhov[ip], E[ip], &fx_rho1, &fx_rhou1, &fx_rhov1, &fx_E1);
            fluxX(rho[im], rhou[im], rhov[im], E[im], &fx_rho2, &fx_rhou2, &fx_rhov2, &fx_E2);
            fluxY(rho[jp], rhou[jp], rhov[jp], E[jp], &fy_rho1, &fy_rhou1, &fy_rhov1, &fy_E1);
            fluxY(rho[jm], rhou[jm], rhov[jm], E[jm], &fy_rho2, &fy_rhou2, &fy_rhov2, &fy_E2);

            rho_new[c] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);
            rhou_new[c] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);
            rhov_new[c] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);
            E_new[c] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            const int c = idx(i, j, stride);
            rho[c] = rho_new[c];
            rhou[c] = rhou_new[c];
            rhov[c] = rhov_new[c];
            E[c] = E_new[c];
        }
    }

    double total_kinetic = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:total_kinetic)
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            const int c = idx(i, j, stride);
            double u = rhou[c] / rho[c];
            double v = rhov[c] / rho[c];
            total_kinetic += 0.5 * rho[c] * (u * u + v * v);
        }
    }
    return total_kinetic;
}

static RunResult run_cpu(const Config &cfg, double rho0, double u0, double v0, double E0,
                         double dt, double dx, double dy, const bool *solid,
                         double *rho, double *rhou, double *rhov, double *E,
                         double *rho_new, double *rhou_new, double *rhov_new, double *E_new) {
    RunResult result;
    auto start = chrono::steady_clock::now();
    for (int n = 0; n < cfg.nSteps; n++) {
        apply_cpu_boundary_conditions(cfg.Nx, cfg.Ny, rho0, u0, v0, E0, rho, rhou, rhov, E);
        result.total_kinetic = update_cpu(cfg.Nx, cfg.Ny, dt, dx, dy, solid,
                                          rho, rhou, rhov, E, rho_new, rhou_new, rhov_new, E_new);
        if (cfg.progress && n % 50 == 0) {
            cout << "Step " << n << " completed, total kinetic energy: " << result.total_kinetic << '\n';
        }
    }
    auto end = chrono::steady_clock::now();
    result.seconds = chrono::duration<double>(end - start).count();
    return result;
}

static RunResult run_gpu(const Config &cfg, double rho0, double u0, double v0, double E0,
                         double dt, double dx, double dy, const bool *solid,
                         double *rho, double *rhou, double *rhov, double *E,
                         double *rho_new, double *rhou_new, double *rhov_new, double *E_new) {
    const int Nx = cfg.Nx;
    const int Ny = cfg.Ny;
    const int stride = Ny + 2;
    const int total_size = (Nx + 2) * (Ny + 2);
    const double dtdx = dt / (2.0 * dx);
    const double dtdy = dt / (2.0 * dy);
    RunResult result;

    auto start = chrono::steady_clock::now();
    #pragma omp target data map(tofrom: rho[0:total_size], rhou[0:total_size], rhov[0:total_size], E[0:total_size]) \
                            map(alloc: rho_new[0:total_size], rhou_new[0:total_size], rhov_new[0:total_size], E_new[0:total_size]) \
                            map(to: solid[0:total_size])
    {
        for (int n = 0; n < cfg.nSteps; n++) {
            #pragma omp target teams distribute parallel for
            for (int j = 0; j < Ny + 2; j++) {
                rho[idx(0, j, stride)] = rho0;
                rhou[idx(0, j, stride)] = rho0 * u0;
                rhov[idx(0, j, stride)] = rho0 * v0;
                E[idx(0, j, stride)] = E0;

                rho[idx(Nx + 1, j, stride)] = rho[idx(Nx, j, stride)];
                rhou[idx(Nx + 1, j, stride)] = rhou[idx(Nx, j, stride)];
                rhov[idx(Nx + 1, j, stride)] = rhov[idx(Nx, j, stride)];
                E[idx(Nx + 1, j, stride)] = E[idx(Nx, j, stride)];
            }

            #pragma omp target teams distribute parallel for
            for (int i = 0; i < Nx + 2; i++) {
                rho[idx(i, 0, stride)] = rho[idx(i, 1, stride)];
                rhou[idx(i, 0, stride)] = rhou[idx(i, 1, stride)];
                rhov[idx(i, 0, stride)] = -rhov[idx(i, 1, stride)];
                E[idx(i, 0, stride)] = E[idx(i, 1, stride)];

                rho[idx(i, Ny + 1, stride)] = rho[idx(i, Ny, stride)];
                rhou[idx(i, Ny + 1, stride)] = rhou[idx(i, Ny, stride)];
                rhov[idx(i, Ny + 1, stride)] = -rhov[idx(i, Ny, stride)];
                E[idx(i, Ny + 1, stride)] = E[idx(i, Ny, stride)];
            }

            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    const int c = idx(i, j, stride);
                    if (solid[c]) {
                        rho_new[c] = rho[c];
                        rhou_new[c] = rhou[c];
                        rhov_new[c] = rhov[c];
                        E_new[c] = E[c];
                    } else {
                        const int ip = idx(i + 1, j, stride);
                        const int im = idx(i - 1, j, stride);
                        const int jp = idx(i, j + 1, stride);
                        const int jm = idx(i, j - 1, stride);

                        rho_new[c] = 0.25 * (rho[ip] + rho[im] + rho[jp] + rho[jm]);
                        rhou_new[c] = 0.25 * (rhou[ip] + rhou[im] + rhou[jp] + rhou[jm]);
                        rhov_new[c] = 0.25 * (rhov[ip] + rhov[im] + rhov[jp] + rhov[jm]);
                        E_new[c] = 0.25 * (E[ip] + E[im] + E[jp] + E[jm]);

                        double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                        double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                        double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                        double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                        fluxX(rho[ip], rhou[ip], rhov[ip], E[ip], &fx_rho1, &fx_rhou1, &fx_rhov1, &fx_E1);
                        fluxX(rho[im], rhou[im], rhov[im], E[im], &fx_rho2, &fx_rhou2, &fx_rhov2, &fx_E2);
                        fluxY(rho[jp], rhou[jp], rhov[jp], E[jp], &fy_rho1, &fy_rhou1, &fy_rhov1, &fy_E1);
                        fluxY(rho[jm], rhou[jm], rhov[jm], E[jm], &fy_rho2, &fy_rhou2, &fy_rhov2, &fy_E2);

                        rho_new[c] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);
                        rhou_new[c] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);
                        rhov_new[c] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);
                        E_new[c] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);
                    }
                }
            }

            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    const int c = idx(i, j, stride);
                    rho[c] = rho_new[c];
                    rhou[c] = rhou_new[c];
                    rhov[c] = rhov_new[c];
                    E[c] = E_new[c];
                }
            }

            double total_kinetic = 0.0;
            #pragma omp target teams distribute parallel for collapse(2) reduction(+:total_kinetic) map(tofrom: total_kinetic)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    const int c = idx(i, j, stride);
                    double u = rhou[c] / rho[c];
                    double v = rhov[c] / rho[c];
                    total_kinetic += 0.5 * rho[c] * (u * u + v * v);
                }
            }
            result.total_kinetic = total_kinetic;

            if (cfg.progress && n % 50 == 0) {
                cout << "Step " << n << " completed, total kinetic energy: " << result.total_kinetic << '\n';
            }
        }
    }
    auto end = chrono::steady_clock::now();
    result.seconds = chrono::duration<double>(end - start).count();
    return result;
}

static RunResult run_simulation(const Config &cfg) {
    const int Nx = cfg.Nx;
    const int Ny = cfg.Ny;
    const double Lx = 2.0;
    const double Ly = 1.0;
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;
    const int total_size = (Nx + 2) * (Ny + 2);
    const int stride = Ny + 2;

    double *rho = static_cast<double *>(malloc(total_size * sizeof(double)));
    double *rhou = static_cast<double *>(malloc(total_size * sizeof(double)));
    double *rhov = static_cast<double *>(malloc(total_size * sizeof(double)));
    double *E = static_cast<double *>(malloc(total_size * sizeof(double)));
    double *rho_new = static_cast<double *>(malloc(total_size * sizeof(double)));
    double *rhou_new = static_cast<double *>(malloc(total_size * sizeof(double)));
    double *rhov_new = static_cast<double *>(malloc(total_size * sizeof(double)));
    double *E_new = static_cast<double *>(malloc(total_size * sizeof(double)));
    bool *solid = static_cast<bool *>(malloc(total_size * sizeof(bool)));

    if (!rho || !rhou || !rhov || !E || !rho_new || !rhou_new || !rhov_new || !E_new || !solid) {
        cerr << "Allocation failed for grid " << Nx << "x" << Ny << '\n';
        exit(2);
    }

    const double cx = 0.5;
    const double cy = 0.5;
    const double radius = 0.1;

    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0 / (gamma_val - 1.0) + 0.5 * rho0 * (u0 * u0 + v0 * v0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx + 2; i++) {
        for (int j = 0; j < Ny + 2; j++) {
            const int c = idx(i, j, stride);
            const double x = (i - 0.5) * dx;
            const double y = (j - 0.5) * dy;
            const bool is_solid = (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius;
            solid[c] = is_solid;
            rho[c] = rho0;
            rhou[c] = is_solid ? 0.0 : rho0 * u0;
            rhov[c] = is_solid ? 0.0 : rho0 * v0;
            E[c] = is_solid ? p0 / (gamma_val - 1.0) : E0;
            rho_new[c] = 0.0;
            rhou_new[c] = 0.0;
            rhov_new[c] = 0.0;
            E_new[c] = 0.0;
        }
    }

    double c0 = sqrt(gamma_val * p0 / rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0) / 2.0;

    RunResult result;
    if (cfg.backend == Backend::Gpu) {
        result = run_gpu(cfg, rho0, u0, v0, E0, dt, dx, dy, solid,
                         rho, rhou, rhov, E, rho_new, rhou_new, rhov_new, E_new);
    } else {
        result = run_cpu(cfg, rho0, u0, v0, E0, dt, dx, dy, solid,
                         rho, rhou, rhov, E, rho_new, rhou_new, rhov_new, E_new);
    }

    free(rho);
    free(rhou);
    free(rhov);
    free(E);
    free(rho_new);
    free(rhou_new);
    free(rhov_new);
    free(E_new);
    free(solid);

    return result;
}

int main(int argc, char **argv) {
    Config cfg;
    if (!parse_args(argc, argv, &cfg)) {
        return 1;
    }

    RunResult result = run_simulation(cfg);
    cout << fixed << setprecision(6)
         << "backend,Nx,Ny,nSteps,runtime_seconds,total_kinetic\n"
         << (cfg.backend == Backend::Gpu ? "gpu" : "cpu") << ','
         << cfg.Nx << ',' << cfg.Ny << ',' << cfg.nSteps << ','
         << result.seconds << ',' << setprecision(12) << result.total_kinetic << '\n';

    return 0;
}
