#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace pctplanner {

struct TravParams {
    double resolution;
    double slice_dh;
    double interval_min;
    double interval_free;
    double slope_max;
    double step_max;
    double standable_ratio;
    double cost_barrier;
    double safe_margin;
    double inflation;
    int kernel_size;
};

struct GpuTomographyOutput {
    std::vector<float> trav_cost;       // raw traversability cost
    std::vector<float> inflated_cost;   // inflated traversability
    std::vector<float> interval;        // ceiling - ground
    std::vector<float> grad_mag_sq;     // gradient magnitude squared on ground
    std::vector<float> grad_mag_max;    // per-cell max gradient component on ground
    std::vector<float> elev_g;          // ground height per slice cell
    std::vector<float> elev_c;          // ceiling height per slice cell
};

bool RunTomographyCuda(const std::vector<Eigen::Vector3f>& points,
                       uint32_t n_slice, uint32_t dim_x, uint32_t dim_y,
                       float cx, float cy, float slice_h0,
                       const TravParams& params,
                       GpuTomographyOutput& out);

}  // namespace pctplanner
