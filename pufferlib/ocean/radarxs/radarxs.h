#define _GNU_SOURCE
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NO_RAYLIB
#include "raylib.h"

/////////////////////////////////////////////////////////////
// Const Dump Start

const Color color_bgteal = (Color){6, 24, 24, 255}; // Darkteal
const Color color_blue = (Color){0, 0, 255, 255}; // Blue
const Color color_sky = (Color){0, 128, 255, 255}; // Sky
const Color color_gray = (Color){128, 128, 128, 255}; // Gray
const Color color_red = (Color){255, 0, 0, 255}; // Red
const Color color_white = (Color){255, 255, 255, 255}; // White
const Color color_salmon = (Color){255, 85, 85, 255}; // Salmon
const Color color_silver = (Color){170, 170, 170, 255}; // Silver
const Color color_cyan = (Color){0, 255, 255, 255}; // Cyan
const Color color_yellow = (Color){255, 255, 0, 255}; // Yellow
#endif

const bool WRITE_LOGS_TO_FILES = false;
const unsigned char SEARCH = 0;
const unsigned char TRACK1 = 1;
const unsigned char TRACK2 = 2;
const unsigned char TRACK3 = 3;
const unsigned char TRACK4 = 4;
const unsigned char TRACK5 = 5;
const unsigned char NOOP = 6;

#define MAX_AZ_SLICES 30
#define MAX_EL_SLICES 10
const float AZ_DEGREES_PER_SLICE = 90.0f / MAX_AZ_SLICES;
const float EL_DEGREES_PER_SLICE = 30.0f / MAX_EL_SLICES;

const int MAX_SEARCHERS = 1;
const int FEATURES_PER_TRACKER = 6;  // t_desired, t_deadline, t_dwell_estimate, priority, az_bin, el_bin

const int PLACEHOLDER_FOR_SENSOR_ID = 1;

const float S_BAND_MAX_RANGE = 184000000.0f;  // 184 km in millimeters
const float X_BAND_MAX_RANGE = 100000000.0f;  // 100 km in millimeters
const float S_BAND_MIN_RANGE = 10000000.0f;   // 10 km in millimeters
const float X_BAND_MIN_RANGE = 5000000.0f;    // 5 km in millimeters

const float MAX_TARGET_XY_RANGE = 184000000.0f;  // 184 km in millimeters
const float MAX_TARGET_Z_RANGE = 20000000.0f;    // 20 km in millimeters
const float MAX_TARGET_XY_VELOCITY = 1000.0f;    // 1000 m/s
const float MIN_TARGET_SINGER_SIGMA = 0.0f;
const float MAX_TARGET_SINGER_SIGMA = 35.0f;
const float MIN_TARGET_SINGER_THETA = 1.0f;   // TODO: should this be 1000 milliseconds?
const float MAX_TARGET_SINGER_THETA = 50.0f;  // TODO: should this be 50000 milliseconds?
// Table III "Singer Manoeuvre Parameters for Three Target Types"
// From A. Charlish, K. Woodbridge, and H. Griffiths,
// â€˜Phased array radar resource management using continuous double auctionâ€™,
// IEEE Transactions on Aerospace and Electronic Systems,
// vol. 51, no. 3, pp. 2212â€“2224, 2015.
// DOI. No. 10.1109/TAES.2015.130558.
// Type 1: BigSigma in [20,35], BigTheta in [10,20]
// Type 2: BigSigma in [0,5], BigTheta in [1,4]
// Type 3: BigSigma in [5,20], BigTheta in [30,50]
// For now I'm just doing (min, max) over all types.

const int PRIORITY_LEVELS = 3;

const int MIN_DWELL_TIME = 10;     // 10 milliseconds
const int MAX_DWELL_TIME = 200;    // 100 milliseconds
const int MAX_DEADLINE = 30000;     // 30000 milliseconds
const int MIN_REVISIT_TIME = 100;  // 100 milliseconds (page 192 of VKB)
#define SEARCH_DWELL_TIME 10  // 10 milliseconds (from prior radar scheduling setup)

// For reset
#define ZERO_COST_SEARCH_TIME (MAX_AZ_SLICES * MAX_EL_SLICES * SEARCH_DWELL_TIME)
const int NO_TARGET = -1;

const unsigned int S_BAND_SENSOR = 0;
const unsigned int X_BAND_SENSOR = 1;
const int SENSOR_IMPLICIT = -1;
const int SENSOR_S_BAND = 0;
const int SENSOR_X_BAND = 1;

const float REFERENCE_DWELL_TIME = 10.0f;  //  10 milliseconds
const float REFERENCE_RANGE = 184000000.0f;
const float REFERENCE_CROSS_SECTION = 1.0f;
const float REFERENCE_SNR = 40.0f;

const float TRACK_UPDATE_REWARD = 0.1f;
const float SEARCH_ACTION_REWARD = 0.1f;
const float TRACK_LOSS_PENALTY = 1.0f; // Default coefficient by priority
const float TRACK_DELAY_PENALTY = 1.0f/1000.0f;  // Penalty per second to ms
const float GLOBAL_DELAY_PENALTY = 1.0f/1000.0f; // Per-ms overdue increment penalty
const float SEARCH_PENALTY = 0.1f/1000.0f; // Penalty per second to ms
const int SEARCH_DELAY_MODE = 0;  // 0=linear, 1=exponential
const float SEARCH_DEBT_PENALTY_WEIGHT = 0.1f/1000.0f;
const float SEARCH_DEBT_TAU_MS = 10.0f;
const float SEARCH_DELAY_PENALTY_CAP = -1.0f;
const int SECTOR_STALENESS_MODE = 0;  // 0=linear, 1=exponential
const float SECTOR_STALENESS_TAU_MS = 500.0f;

const float VERTICAL_MOTION_FACTOR = 0.1f;

const int WINDOW_X_PX = 480;
const int WINDOW_Y_PX = 270;

// Const Dump End
/////////////////////////////////////////////////////////////

typedef struct {
  float x;
  float x_velocity;
  float x_acceleration;
  float y;
  float y_velocity;
  float y_acceleration;
  float z;
  float z_velocity;
  float z_acceleration;
  float singer_sigma;  // maneuver standard deviation
  float singer_theta;  // maneuver time constant
  float priority;
  float latent_t_desired;
  float latent_t_deadline;
  float latent_t_dwell_estimate;
  float service_age_ms;
  bool is_active;
  bool is_tracked;
} Target;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;


// Required struct named same as env 
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int tick;
    int s_band_t_until_free;
    int x_band_t_until_free;
    Target *targets;  // should I allow more targets than trackers?
    int initial_targets;
    int max_trackers;
    int enable_global_delay;
    int enable_local_delay;
    int enable_x_band;
    int enable_search_refresh_tracked;
    float search_refresh_gain;
    int enable_priority;
    int enable_poisson_arrivals;
    int activate_all_targets_without_poisson;
    float poisson_rate_per_second;
    float search_action_reward;
    float track_update_reward;
    float track_loss_penalty;
    float track_urgency_bonus_weight;
    float target_service_weight;
    float target_service_horizon_ms;
    float sector_staleness_weight;
    float searched_sector_reward_weight;
    float search_frame_overdue_weight;
    float search_frame_desired_ms;
    float search_frame_deadline_ms;
    float search_frame_drop_penalty;
    int search_task_cost_mode;
    float revisit_time_scale;
    float dwell_time_scale;
    int penalize_hidden_targets;
    int enable_track_beam_scan;
    int episode_time_limit_ms;
    float search_debt_ms;
    int search_delay_mode;
    float search_debt_penalty_weight;
    float search_debt_tau_ms;
    float search_delay_penalty_cap;
    unsigned int rng_state;
} Radarxs;

static inline unsigned int radarxs_rng_next(Radarxs *env) {
  env->rng_state = 1664525u * env->rng_state + 1013904223u;
  return env->rng_state;
}

static inline float radarxs_uniform01(Radarxs *env) {
  // Use the upper 24 bits to produce a reproducible float in [0, 1).
  return (float)((radarxs_rng_next(env) >> 8) & 0x00FFFFFFu) / 16777216.0f;
}

static inline int radarxs_randint(Radarxs *env, int n) {
  if (n <= 1) {
    return 0;
  }
  return (int)(radarxs_rng_next(env) % (unsigned int)n);
}

static inline float search_delay_penalty(float debt_ms, int mode, float weight, float tau_ms, float cap) {
  float penalty;
  if (weight <= 0.0f || debt_ms <= 0.0f) {
    return 0.0f;
  }
  if (mode == 0) {
    penalty = weight * debt_ms;
  } else {
    float arg = debt_ms / fmaxf(1e-3f, tau_ms);
    if (arg > 20.0f) {
      arg = 20.0f;
    }
    penalty = weight * (expf(arg) - 1.0f);
  }
  if (cap >= 0.0f && penalty > cap) {
    penalty = cap;
  }
  return penalty;
}

static inline float sector_staleness_penalty(float mean_stale_before, float mean_stale_after, float weight) {
  if (weight <= 0.0f) {
    return 0.0f;
  }
  if (SECTOR_STALENESS_MODE == 0) {
    // Potential-based surveillance reward. Callers subtract this value:
    // increasing stale debt is a penalty, decreasing stale debt is a reward.
    return weight * (mean_stale_after - mean_stale_before);
  }
  float before_arg = mean_stale_before / SECTOR_STALENESS_TAU_MS;
  float after_arg = mean_stale_after / SECTOR_STALENESS_TAU_MS;
  if (before_arg > 20.0f) before_arg = 20.0f;
  if (after_arg > 20.0f) after_arg = 20.0f;
  return weight * (
      (expf(after_arg) - 1.0f) -
      (expf(before_arg) - 1.0f)
  );
}

int sample_poisson(Radarxs *env, float lambda) {
  float L;
  float p;
  int k;

  if (lambda <= 0.0f) {
    return 0;
  }

  L = expf(-lambda);
  k = 0;
  p = 1.0f;
  do {
    k++;
    p *= radarxs_uniform01(env);
  } while (p > L);
  return k - 1;
}

void compute_tracker_timers(Radarxs *env, int tracker_id, float *t_desired_out,
                            float *t_deadline_out, float *t_dwell_out,
                            float *az_norm_out, float *el_norm_out);
void write_tracker_observation(Radarxs *env, int tracker_id, float t_desired,
                               float t_deadline, float t_dwell_estimate,
                               float az_norm, float el_norm);

void activate_inactive_target(Radarxs *env, int target_index) {
  float az_norm = 0.0f;
  float el_norm = 0.0f;
  env->targets[target_index].is_active = true;
  env->targets[target_index].is_tracked = false;
  compute_tracker_timers(env, target_index,
                         &env->targets[target_index].latent_t_desired,
                         &env->targets[target_index].latent_t_deadline,
                         &env->targets[target_index].latent_t_dwell_estimate,
                         &az_norm, &el_norm);
}

void maybe_spawn_poisson_arrivals(Radarxs *env, int delta_t) {
  int arrivals;
  int activated;

  if (!env->enable_poisson_arrivals || delta_t <= 0 || env->poisson_rate_per_second <= 0.0f) {
    return;
  }

  arrivals = sample_poisson(env, env->poisson_rate_per_second * ((float)delta_t / 1000.0f));
  activated = 0;
  for (int i = 0; i < env->max_trackers && activated < arrivals; i++) {
    if (!env->targets[i].is_active) {
      activate_inactive_target(env, i);
      activated++;
    }
  }
}

void initialize_target(Radarxs *env, int target_index) {
  int base_idx;
  env->targets[target_index].x = radarxs_uniform01(env) * MAX_TARGET_XY_RANGE;
  env->targets[target_index].y = radarxs_uniform01(env) * MAX_TARGET_XY_RANGE;
  env->targets[target_index].z = radarxs_uniform01(env) * MAX_TARGET_Z_RANGE;

  while (
    sqrt(
    env->targets[target_index].x * env->targets[target_index].x +
    env->targets[target_index].y * env->targets[target_index].y +
    env->targets[target_index].z * env->targets[target_index].z) >
    S_BAND_MAX_RANGE
    ) {
    env->targets[target_index].x = radarxs_uniform01(env) * MAX_TARGET_XY_RANGE;
    env->targets[target_index].y = radarxs_uniform01(env) * MAX_TARGET_XY_RANGE;
    env->targets[target_index].z = radarxs_uniform01(env) * MAX_TARGET_Z_RANGE;
  }

  
  env->targets[target_index].x_velocity =
      radarxs_uniform01(env) * MAX_TARGET_XY_VELOCITY;
  env->targets[target_index].y_velocity =
      radarxs_uniform01(env) * MAX_TARGET_XY_VELOCITY;
  env->targets[target_index].z_velocity = 0;  // targets spawn doing level flight
  env->targets[target_index].x_acceleration = 0;
  env->targets[target_index].y_acceleration = 0;
  env->targets[target_index].z_acceleration = 0;
  env->targets[target_index].singer_sigma =
      radarxs_uniform01(env) * (MAX_TARGET_SINGER_SIGMA - MIN_TARGET_SINGER_SIGMA) +
      MIN_TARGET_SINGER_SIGMA;
  env->targets[target_index].singer_theta =
      radarxs_uniform01(env) * (MAX_TARGET_SINGER_THETA - MIN_TARGET_SINGER_THETA) +
      MIN_TARGET_SINGER_THETA;
  env->targets[target_index].priority =
      env->enable_priority ? radarxs_randint(env, PRIORITY_LEVELS) : 0.0f;
  env->targets[target_index].latent_t_desired = NO_TARGET;
  env->targets[target_index].latent_t_deadline = NO_TARGET;
  env->targets[target_index].latent_t_dwell_estimate = 0.0f;
  env->targets[target_index].is_active =
      env->enable_poisson_arrivals ? false : env->activate_all_targets_without_poisson;
  env->targets[target_index].is_tracked = false;

  base_idx = MAX_AZ_SLICES * MAX_EL_SLICES + target_index * FEATURES_PER_TRACKER;
  env->observations[base_idx] = NO_TARGET;
  env->observations[base_idx + 1] = NO_TARGET;
  env->observations[base_idx + 2] = 0.0f;
  env->observations[base_idx + 3] = env->targets[target_index].priority;
  env->observations[base_idx + 4] = 0.0f;
  env->observations[base_idx + 5] = 0.0f;
}

float normal(Radarxs *env, float mean, float stddev) {
  float u1 = fmaxf(radarxs_uniform01(env), 1e-7f);
  float u2 = radarxs_uniform01(env);
  // Apply Box-Muller transform
  float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
  return mean + stddev * z0;
}

void compute_tracker_timers(Radarxs *env, int tracker_id, float *t_desired_out,
                            float *t_deadline_out, float *t_dwell_out,
                            float *az_norm_out, float *el_norm_out) {
  float target_range = sqrt(env->targets[tracker_id].x * env->targets[tracker_id].x +
                            env->targets[tracker_id].y * env->targets[tracker_id].y +
                            env->targets[tracker_id].z * env->targets[tracker_id].z);
  float az = 0.0f;
  float el = 0.0f;
  int az_idx = 0;
  int el_idx = 0;
  float target_cross_section = 4.2f;
  float sigma_theta = 1;
  float u = 0.3f;

  float new_t_desired =
      1000.0f *
      (0.4f *
       pow((target_range / 1000.0f * sigma_theta * sqrt(env->targets[tracker_id].singer_theta) /
            env->targets[tracker_id].singer_sigma),
           0.4f) *
       pow(u, 2.4f) / (1 + 0.5f * pow(u, 2)));
  if (env->enable_priority) {
    float revisit_scale = 1.0f - 0.25f * env->targets[tracker_id].priority;
    if (revisit_scale < 0.35f) {
      revisit_scale = 0.35f;
    }
    new_t_desired *= revisit_scale;
  }
  new_t_desired *= env->revisit_time_scale;
  if (new_t_desired < MIN_REVISIT_TIME) {
    new_t_desired = MIN_REVISIT_TIME;
  }
  if (new_t_desired > MAX_DEADLINE) {
    new_t_desired = MAX_DEADLINE;
  }
  float new_t_deadline =
      new_t_desired * (2.5f - 0.75f * env->targets[tracker_id].priority);
  if (new_t_deadline < new_t_desired) {
    new_t_deadline = new_t_desired;
  }
  if (new_t_deadline > MAX_DEADLINE) {
    new_t_deadline = MAX_DEADLINE;
  }
  float new_t_dwell_estimate =
      (REFERENCE_DWELL_TIME * pow((target_range / REFERENCE_RANGE), 4) *
       (REFERENCE_CROSS_SECTION / target_cross_section) * REFERENCE_SNR);
  new_t_dwell_estimate *= env->dwell_time_scale;
  if (new_t_dwell_estimate < MIN_DWELL_TIME) {
    new_t_dwell_estimate = MIN_DWELL_TIME;
  }
  if (new_t_dwell_estimate > MAX_DWELL_TIME) {
    new_t_dwell_estimate = MAX_DWELL_TIME;
  }

  if (target_range > 0.0f) {
    az = atan2(env->targets[tracker_id].y, env->targets[tracker_id].x);
    el = asin(env->targets[tracker_id].z / target_range);
    az_idx = (int)(az * 180.0f / M_PI / AZ_DEGREES_PER_SLICE);
    el_idx = (int)(el * 180.0f / M_PI / EL_DEGREES_PER_SLICE);
    if (az_idx < 0) az_idx = 0;
    if (az_idx >= MAX_AZ_SLICES) az_idx = MAX_AZ_SLICES - 1;
    if (el_idx < 0) el_idx = 0;
    if (el_idx >= MAX_EL_SLICES) el_idx = MAX_EL_SLICES - 1;
  }

  *t_desired_out = new_t_desired;
  *t_deadline_out = new_t_deadline;
  *t_dwell_out = new_t_dwell_estimate;
  *az_norm_out = (target_range > 0.0f && MAX_AZ_SLICES > 1) ? ((float)az_idx / (float)(MAX_AZ_SLICES - 1)) : 0.0f;
  *el_norm_out = (target_range > 0.0f && MAX_EL_SLICES > 1) ? ((float)el_idx / (float)(MAX_EL_SLICES - 1)) : 0.0f;
}

void write_tracker_observation(Radarxs *env, int tracker_id, float t_desired,
                               float t_deadline, float t_dwell_estimate,
                               float az_norm, float el_norm) {
  int base = MAX_AZ_SLICES * MAX_EL_SLICES + tracker_id * FEATURES_PER_TRACKER;
  env->observations[base] = t_desired;
  env->observations[base + 1] = t_deadline;
  env->observations[base + 2] = t_dwell_estimate;
  env->observations[base + 3] = env->targets[tracker_id].priority;
  env->observations[base + 4] = az_norm;
  env->observations[base + 5] = el_norm;
}

void update_tracker(Radarxs *env, int tracker_id) {
  float new_t_desired;
  float new_t_deadline;
  float new_t_dwell_estimate;
  float az_norm;
  float el_norm;
  compute_tracker_timers(env, tracker_id, &new_t_desired, &new_t_deadline,
                         &new_t_dwell_estimate, &az_norm, &el_norm);
  if (WRITE_LOGS_TO_FILES) {
    FILE *file = fopen("./logs/new_t_desired.csv", "a");
    fprintf(file, "%d,%d,%f\n", env->tick, tracker_id, new_t_desired);
    fclose(file);
  }
  env->targets[tracker_id].latent_t_desired = new_t_desired;
  env->targets[tracker_id].latent_t_deadline = new_t_deadline;
  env->targets[tracker_id].latent_t_dwell_estimate = new_t_dwell_estimate;
  env->targets[tracker_id].service_age_ms = 0.0f;
  write_tracker_observation(env, tracker_id, new_t_desired, new_t_deadline,
                            new_t_dwell_estimate, az_norm, el_norm);
}

void update_tracker_partial(Radarxs *env, int tracker_id, float gain) {
  int base_idx = MAX_AZ_SLICES * MAX_EL_SLICES + tracker_id * FEATURES_PER_TRACKER;
  float old_t_desired = env->observations[base_idx];
  float old_t_deadline = env->observations[base_idx + 1];
  float old_t_dwell = env->observations[base_idx + 2];

  if (gain <= 0.0f) {
    return;
  }
  if (gain >= 1.0f) {
    update_tracker(env, tracker_id);
    return;
  }

  update_tracker(env, tracker_id);
  env->observations[base_idx] =
      old_t_desired + gain * (env->observations[base_idx] - old_t_desired);
  env->observations[base_idx + 1] =
      old_t_deadline + gain * (env->observations[base_idx + 1] - old_t_deadline);
  env->observations[base_idx + 2] =
      old_t_dwell + gain * (env->observations[base_idx + 2] - old_t_dwell);
  env->targets[tracker_id].latent_t_desired = env->observations[base_idx];
  env->targets[tracker_id].latent_t_deadline = env->observations[base_idx + 1];
  env->targets[tracker_id].latent_t_dwell_estimate = env->observations[base_idx + 2];
}

float get_target_range(Radarxs *env, int target_index) {
  return sqrt(env->targets[target_index].x * env->targets[target_index].x +
              env->targets[target_index].y * env->targets[target_index].y +
              env->targets[target_index].z * env->targets[target_index].z);
}

float get_detection_probability(Radarxs *env, float target_range) {
  if (env->s_band_t_until_free == 0) {
    if (target_range >= S_BAND_MIN_RANGE && target_range <= S_BAND_MAX_RANGE) {
      return 1 - exp((((-10e32 / target_range) / target_range) / target_range) / target_range);
    }
  } else {
    if (target_range >= X_BAND_MIN_RANGE && target_range <= X_BAND_MAX_RANGE) {
      return 1 - exp((((-10e31 / target_range) / target_range) / target_range) / target_range);
    }
  }
  return 0.0f;
}

bool target_is_in_sector(Radarxs *env, int target_index, int sector, float max_range) {
  int az_idx = sector % MAX_AZ_SLICES;
  int el_idx = sector / MAX_AZ_SLICES;
  float az_min = az_idx * (AZ_DEGREES_PER_SLICE) * M_PI / 180.0f;
  float az_max = (az_idx + 1) * (AZ_DEGREES_PER_SLICE) * M_PI / 180.0f;
  float el_min = el_idx * (EL_DEGREES_PER_SLICE) * M_PI / 180.0f;
  float el_max = (el_idx + 1) * (EL_DEGREES_PER_SLICE) * M_PI / 180.0f;
  float target_range = get_target_range(env, target_index);
  float az;
  float el;

  if (target_range <= 0.0f || target_range > max_range) {
    return false;
  }

  az = atan2(env->targets[target_index].y, env->targets[target_index].x);
  el = asin(env->targets[target_index].z / target_range);
  return az >= az_min && az < az_max && el >= el_min && el < el_max;
}

int target_sector(Radarxs *env, int target_index) {
  float target_range = get_target_range(env, target_index);
  float az;
  float el;
  int az_idx;
  int el_idx;

  if (target_range <= 0.0f) {
    return -1;
  }

  az = atan2(env->targets[target_index].y, env->targets[target_index].x);
  el = asin(env->targets[target_index].z / target_range);
  az_idx = (int)(az * 180.0f / M_PI / AZ_DEGREES_PER_SLICE);
  el_idx = (int)(el * 180.0f / M_PI / EL_DEGREES_PER_SLICE);

  if (az_idx < 0) az_idx = 0;
  if (az_idx >= MAX_AZ_SLICES) az_idx = MAX_AZ_SLICES - 1;
  if (el_idx < 0) el_idx = 0;
  if (el_idx >= MAX_EL_SLICES) el_idx = MAX_EL_SLICES - 1;

  return el_idx * MAX_AZ_SLICES + az_idx;
}

void scan_sector(Radarxs *env, int sector, float max_range) {
  for (int i = 0; i < env->max_trackers; i++) {
    float target_range;
    float probability_of_detection;

    if (!env->targets[i].is_active) {
      continue;
    }

    if (!target_is_in_sector(env, i, sector, max_range)) {
      continue;
    }

    if (env->targets[i].is_tracked) {
      if (!env->enable_search_refresh_tracked) {
        continue;
      }
      update_tracker_partial(env, i, env->search_refresh_gain);
      continue;
    }

    target_range = get_target_range(env, i);
    probability_of_detection = get_detection_probability(env, target_range);
    if (radarxs_uniform01(env) < probability_of_detection) {
      float az_norm = 0.0f;
      float el_norm = 0.0f;
      env->targets[i].is_tracked = true;
      if (env->targets[i].latent_t_desired == NO_TARGET ||
          env->targets[i].latent_t_deadline == NO_TARGET) {
        compute_tracker_timers(env, i,
                               &env->targets[i].latent_t_desired,
                               &env->targets[i].latent_t_deadline,
                               &env->targets[i].latent_t_dwell_estimate,
                               &az_norm, &el_norm);
      } else {
        int detected_sector = target_sector(env, i);
        if (detected_sector >= 0) {
          int az_idx = detected_sector % MAX_AZ_SLICES;
          int el_idx = detected_sector / MAX_AZ_SLICES;
          az_norm = (MAX_AZ_SLICES > 1) ? ((float)az_idx / (float)(MAX_AZ_SLICES - 1)) : 0.0f;
          el_norm = (MAX_EL_SLICES > 1) ? ((float)el_idx / (float)(MAX_EL_SLICES - 1)) : 0.0f;
        }
      }
      write_tracker_observation(env, i, env->targets[i].latent_t_desired,
                                env->targets[i].latent_t_deadline,
                                env->targets[i].latent_t_dwell_estimate,
                                az_norm, el_norm);
    }
  }
}

void scan_sector_discovery_only(Radarxs *env, int sector, float max_range) {
  for (int i = 0; i < env->max_trackers; i++) {
    float target_range;
    float probability_of_detection;

    if (!env->targets[i].is_active) {
      continue;
    }

    if (!target_is_in_sector(env, i, sector, max_range)) {
      continue;
    }

    if (env->targets[i].is_tracked) {
      continue;
    }

    target_range = get_target_range(env, i);
    probability_of_detection = get_detection_probability(env, target_range);
    if (radarxs_uniform01(env) < probability_of_detection) {
      float az_norm = 0.0f;
      float el_norm = 0.0f;
      env->targets[i].is_tracked = true;
      if (env->targets[i].latent_t_desired == NO_TARGET ||
          env->targets[i].latent_t_deadline == NO_TARGET) {
        compute_tracker_timers(env, i,
                               &env->targets[i].latent_t_desired,
                               &env->targets[i].latent_t_deadline,
                               &env->targets[i].latent_t_dwell_estimate,
                               &az_norm, &el_norm);
      } else {
        int detected_sector = target_sector(env, i);
        if (detected_sector >= 0) {
          int az_idx = detected_sector % MAX_AZ_SLICES;
          int el_idx = detected_sector / MAX_AZ_SLICES;
          az_norm = (MAX_AZ_SLICES > 1) ? ((float)az_idx / (float)(MAX_AZ_SLICES - 1)) : 0.0f;
          el_norm = (MAX_EL_SLICES > 1) ? ((float)el_idx / (float)(MAX_EL_SLICES - 1)) : 0.0f;
        }
      }
      write_tracker_observation(env, i, env->targets[i].latent_t_desired,
                                env->targets[i].latent_t_deadline,
                                env->targets[i].latent_t_dwell_estimate,
                                az_norm, el_norm);
    }
  }
}

void search_sector_with_range(Radarxs *env, int sector, float max_range) {
  scan_sector(env, sector, max_range);

  if (env->searched_sector_reward_weight > 0.0f && env->search_frame_desired_ms > 0.0f) {
    float sector_age = ZERO_COST_SEARCH_TIME - env->observations[sector];
    if (sector_age < 0.0f) sector_age = 0.0f;
    float searched_debt = sector_age / env->search_frame_desired_ms;
    if (searched_debt > 1.0f) searched_debt = 1.0f;
    env->rewards[0] += env->searched_sector_reward_weight * searched_debt;
  }

  if (env->observations[sector] < 0) {
    if (env->sector_staleness_weight > 0.0f) {
      // Potential-based surveillance value: searching a stale sector reduces
      // mean sector debt immediately. The later time-advance block charges the
      // matching positive debt growth as time passes.
      env->rewards[0] += env->sector_staleness_weight *
                         (-env->observations[sector]) /
                         (float)(MAX_AZ_SLICES * MAX_EL_SLICES);
    }
    if (env->searched_sector_reward_weight <= 0.0f &&
        env->search_frame_overdue_weight <= 0.0f) {
      env->rewards[0] += SEARCH_PENALTY * env->observations[sector];
    }
  }
  // Freshness is measured from the end of the emitted beam, not from the start
  // of the action. We therefore offset by the imminent SEARCH_DWELL_TIME so
  // the searched sectors still end this step at ZERO_COST_SEARCH_TIME.
  env->observations[sector] = ZERO_COST_SEARCH_TIME + SEARCH_DWELL_TIME;
}

static inline float target_service_cost(Radarxs *env) {
  if (env->target_service_weight <= 0.0f || env->target_service_horizon_ms <= 0.0f) {
    return 0.0f;
  }
  float cost = 0.0f;
  float horizon = fmaxf(1e-3f, env->target_service_horizon_ms);
  for (int i = 0; i < env->max_trackers; i++) {
    if (env->targets[i].is_active) {
      int base = MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER;
      if (env->observations[base + 1] < 0.0f) {
        continue;
      }
      float pressure = env->targets[i].service_age_ms / horizon;
      if (pressure > 0.0f) {
        float priority_scale = 1.0f + 2.0f * env->targets[i].priority;
        cost += priority_scale * pressure * pressure;
      }
    }
  }
  return env->target_service_weight * cost;
}

void initialize_search_stagger(Radarxs *env) {
  const int macro_rows = MAX_EL_SLICES / 2;
  const int macro_cols = MAX_AZ_SLICES / 2;
  const int macro_count = macro_rows * macro_cols;
  const float macro_step_ms = (macro_count > 0)
      ? ((float)ZERO_COST_SEARCH_TIME / (float)macro_count)
      : (float)ZERO_COST_SEARCH_TIME;

  for (int macro_r = 0; macro_r < macro_rows; macro_r++) {
    for (int macro_c = 0; macro_c < macro_cols; macro_c++) {
      const int macro_idx = macro_r * macro_cols + macro_c;
      const float freshness =
          fmaxf(0.0f, (float)ZERO_COST_SEARCH_TIME - macro_step_ms * (float)macro_idx);
      const int base_r = macro_r * 2;
      const int base_c = macro_c * 2;
      const int sectors[4] = {
          base_c + base_r * MAX_AZ_SLICES,
          base_c + 1 + base_r * MAX_AZ_SLICES,
          base_c + (base_r + 1) * MAX_AZ_SLICES,
          base_c + 1 + (base_r + 1) * MAX_AZ_SLICES,
      };
      for (int k = 0; k < 4; k++) {
        env->observations[sectors[k]] = freshness;
      }
    }
  }
}

void add_log(Radarxs* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(Radarxs *env) {
  env->rng_state = (unsigned int)rand() ^ 0x9E3779B9u;
  if (env->rng_state == 0u) {
    env->rng_state = 1u;
  }
  // Stagger macro-sector freshness across the nominal 3s search cycle.
  // Top-left starts freshest; sectors become progressively older in a
  // structural top-left -> bottom-right order.
  initialize_search_stagger(env);

  // Set all trackers to NO_TARGET
  for (int i = 0; i < env->max_trackers * FEATURES_PER_TRACKER; i++) {
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i] = NO_TARGET;
  }

  // Set the sensor type to S_BAND_SENSOR

  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] =
      S_BAND_SENSOR;

  env->tick = 0;
  env->s_band_t_until_free = 0;
  env->x_band_t_until_free = 0;
  env->search_debt_ms = 0.0f;

  for (int i = 0; i < env->max_trackers; i++) {
    initialize_target(env, i);
  }
  for (int i = 0; i < env->initial_targets; i++) {
    if (env->enable_poisson_arrivals || !env->activate_all_targets_without_poisson) {
      env->targets[i].is_active = true;
    }
    env->targets[i].is_tracked = true;
    update_tracker(env, i);
  }
}


static const int JOINT_ACTION_BASE = 1000000;
static const int JOINT_ACTION_STRIDE = 1000;

static inline void decode_physical_action(Radarxs* env, int raw_action, int* action, int* requested_sensor) {
  *action = raw_action;
  *requested_sensor = SENSOR_IMPLICIT;
  int s_search_action = env->max_trackers + 3;
  int x_search_action = env->max_trackers + 4;
  int s_track_base = env->max_trackers + 5;
  int x_track_base = env->max_trackers + 5 + env->max_trackers;
  if (raw_action == s_search_action) {
    *action = SEARCH;
    *requested_sensor = SENSOR_S_BAND;
  } else if (raw_action == x_search_action) {
    *action = SEARCH;
    *requested_sensor = SENSOR_X_BAND;
  } else if (raw_action >= s_track_base && raw_action < s_track_base + env->max_trackers) {
    *action = (raw_action - s_track_base) + 1;
    *requested_sensor = SENSOR_S_BAND;
  } else if (raw_action >= x_track_base && raw_action < x_track_base + env->max_trackers) {
    *action = (raw_action - x_track_base) + 1;
    *requested_sensor = SENSOR_X_BAND;
  }
}

static inline void dispatch_action_no_advance(Radarxs* env, int action, int requested_sensor) {
  if (action == SEARCH) {
    if (requested_sensor != SENSOR_X_BAND && env->s_band_t_until_free <= 0) {
      int least_recently_used_cluster = 0;
      for (int j = 0; j < MAX_EL_SLICES - 1; j += 2) {
        for (int i = 0; i < MAX_AZ_SLICES - 1; i += 2) {
          int current_cluster_value =
              env->observations[i + j * MAX_AZ_SLICES] +
              env->observations[i + j * MAX_AZ_SLICES + 1] +
              env->observations[i + (j + 1) * MAX_AZ_SLICES] +
              env->observations[i + (j + 1) * MAX_AZ_SLICES + 1];

          int least_cluster_value =
              env->observations[least_recently_used_cluster % MAX_AZ_SLICES +
                                (least_recently_used_cluster / MAX_AZ_SLICES) * MAX_AZ_SLICES] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                (least_recently_used_cluster / MAX_AZ_SLICES) * MAX_AZ_SLICES + 1] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                ((least_recently_used_cluster / MAX_AZ_SLICES) + 1) * MAX_AZ_SLICES] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                ((least_recently_used_cluster / MAX_AZ_SLICES) + 1) * MAX_AZ_SLICES +
                                1];

          if (current_cluster_value < least_cluster_value) {
            least_recently_used_cluster = i + j * MAX_AZ_SLICES;
          }
        }
      }
      search_sector_with_range(env, least_recently_used_cluster, S_BAND_MAX_RANGE);
      search_sector_with_range(env, (least_recently_used_cluster + 1) % (MAX_AZ_SLICES * MAX_EL_SLICES), S_BAND_MAX_RANGE);
      search_sector_with_range(
          env, (least_recently_used_cluster + MAX_AZ_SLICES) % (MAX_AZ_SLICES * MAX_EL_SLICES), S_BAND_MAX_RANGE);
      search_sector_with_range(
          env, (least_recently_used_cluster + MAX_AZ_SLICES + 1) % (MAX_AZ_SLICES * MAX_EL_SLICES), S_BAND_MAX_RANGE);
      env->rewards[0] += env->search_action_reward;
      env->search_debt_ms = 0.0f;
      env->s_band_t_until_free = SEARCH_DWELL_TIME;
    } else if (requested_sensor != SENSOR_S_BAND && env->enable_x_band && env->x_band_t_until_free <= 0) {
      int least_recently_used_sector = 0;
      for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
        if (env->observations[i] < env->observations[least_recently_used_sector]) {
          least_recently_used_sector = i;
        }
      }
      search_sector_with_range(env, least_recently_used_sector, X_BAND_MAX_RANGE);
      env->rewards[0] += 0.25f * env->search_action_reward;
      env->search_debt_ms = 0.0f;
      env->x_band_t_until_free = SEARCH_DWELL_TIME;
    }
  } else if (action <= env->max_trackers) {
    bool emitted_track_beam = false;
    float target_range;
    int covered_sector;

    action -= 1;
    if (!env->targets[action].is_tracked) {
      env->rewards[0] += -env->track_loss_penalty * (1.0f + 2.0f * env->targets[action].priority);
    } else {
      env->rewards[0] += env->track_update_reward;
      {
        float t_des_before =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER];
        float t_dead_before =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER + 1];
        float priority_scale = 1.0f + 2.0f * env->targets[action].priority;
        float overdue_before = (t_des_before < 0.0f) ? -t_des_before : 0.0f;
        float deadline_pressure =
            (t_dead_before < 100.0f) ? (100.0f - t_dead_before) : 0.0f;
        env->rewards[0] += env->track_urgency_bonus_weight *
                           priority_scale *
                           (overdue_before * GLOBAL_DELAY_PENALTY +
                            0.25f * deadline_pressure * GLOBAL_DELAY_PENALTY);
      }
      if (env->enable_local_delay &&
          env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER] < 0) {
          env->rewards[0] += (float)(env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                                                       action * FEATURES_PER_TRACKER] *
                                   TRACK_DELAY_PENALTY * (1 + 2 * env->targets[action].priority));
      }

      target_range = get_target_range(env, action);
      bool use_s_band =
          requested_sensor != SENSOR_X_BAND &&
          env->s_band_t_until_free <= 0 &&
          target_range < S_BAND_MAX_RANGE &&
          target_range > S_BAND_MIN_RANGE;
      bool use_x_band =
          requested_sensor != SENSOR_S_BAND &&
          env->enable_x_band &&
          env->x_band_t_until_free <= 0 &&
          target_range < X_BAND_MAX_RANGE &&
          target_range > X_BAND_MIN_RANGE;
      if (use_s_band || use_x_band) {
        update_tracker(env, action);
        emitted_track_beam = true;
      }

      if (emitted_track_beam && env->enable_track_beam_scan) {
        covered_sector = target_sector(env, action);
        if (covered_sector >= 0) {
          scan_sector_discovery_only(env, covered_sector, target_range);
          env->observations[covered_sector] = ZERO_COST_SEARCH_TIME;
        }
      }

      if (use_s_band) {
        env->s_band_t_until_free =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER + 2];
      } else if (use_x_band) {
        env->x_band_t_until_free =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER + 2] /
            2;
      }
    }
  } else if (action == env->max_trackers + 1 || action == env->max_trackers + 2) {
    env->s_band_t_until_free = SEARCH_DWELL_TIME;
    if (env->enable_x_band) {
      env->x_band_t_until_free = SEARCH_DWELL_TIME;
    }
  }
}

// Required function
void c_step(Radarxs* env) {
  int action = env->actions[0];
  int requested_sensor = SENSOR_IMPLICIT;
  env->terminals[0] = 0;
  env->rewards[0] = 0.0f;
  float target_service_before = target_service_cost(env);

  if (action >= JOINT_ACTION_BASE) {
    int encoded = action - JOINT_ACTION_BASE;
    int s_raw = encoded / JOINT_ACTION_STRIDE;
    int x_raw = encoded % JOINT_ACTION_STRIDE;
    int s_action = s_raw;
    int s_sensor = SENSOR_IMPLICIT;
    int x_action = x_raw;
    int x_sensor = SENSOR_IMPLICIT;
    decode_physical_action(env, s_raw, &s_action, &s_sensor);
    decode_physical_action(env, x_raw, &x_action, &x_sensor);
    if (s_sensor == SENSOR_S_BAND) {
      dispatch_action_no_advance(env, s_action, s_sensor);
    }
    if (x_sensor == SENSOR_X_BAND) {
      dispatch_action_no_advance(env, x_action, x_sensor);
    }
    requested_sensor = SENSOR_IMPLICIT;
    goto move_simulation_forward;
  }

  decode_physical_action(env, action, &action, &requested_sensor);

  if (action == SEARCH) {
    if (requested_sensor != SENSOR_X_BAND && env->s_band_t_until_free <= 0) {
      // Find the stalest non-overlapping 2x2 macro-sector for S-band.
      // Overlapping 2x2 windows create a short raster micro-cycle even for
      // heuristics; we want a structural top-left -> bottom-right sweep.
      int least_recently_used_cluster = 0;
      for (int j = 0; j < MAX_EL_SLICES - 1; j += 2) {
        for (int i = 0; i < MAX_AZ_SLICES - 1; i += 2) {
          int current_cluster_value =
              env->observations[i + j * MAX_AZ_SLICES] +
              env->observations[i + j * MAX_AZ_SLICES + 1] +
              env->observations[i + (j + 1) * MAX_AZ_SLICES] +
              env->observations[i + (j + 1) * MAX_AZ_SLICES + 1];

          int least_cluster_value =
              env->observations[least_recently_used_cluster % MAX_AZ_SLICES +
                                (least_recently_used_cluster / MAX_AZ_SLICES) * MAX_AZ_SLICES] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                (least_recently_used_cluster / MAX_AZ_SLICES) * MAX_AZ_SLICES + 1] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                ((least_recently_used_cluster / MAX_AZ_SLICES) + 1) * MAX_AZ_SLICES] +
              env->observations[(least_recently_used_cluster % MAX_AZ_SLICES) +
                                ((least_recently_used_cluster / MAX_AZ_SLICES) + 1) * MAX_AZ_SLICES +
                                1];

          if (current_cluster_value < least_cluster_value) {
            least_recently_used_cluster = i + j * MAX_AZ_SLICES;
          }
        }
      }

      // S-band searches four sectors, right and below the least recently used sector
      search_sector_with_range(env, least_recently_used_cluster, S_BAND_MAX_RANGE);
      search_sector_with_range(env, (least_recently_used_cluster + 1) % (MAX_AZ_SLICES * MAX_EL_SLICES), S_BAND_MAX_RANGE);
      search_sector_with_range(
          env, (least_recently_used_cluster + MAX_AZ_SLICES) % (MAX_AZ_SLICES * MAX_EL_SLICES), S_BAND_MAX_RANGE);
      search_sector_with_range(
          env, (least_recently_used_cluster + MAX_AZ_SLICES + 1) % (MAX_AZ_SLICES * MAX_EL_SLICES), S_BAND_MAX_RANGE);
      env->rewards[0] += env->search_action_reward;
      env->search_debt_ms = 0.0f;
      env->s_band_t_until_free = SEARCH_DWELL_TIME;
    } else if (requested_sensor != SENSOR_S_BAND && env->enable_x_band && env->x_band_t_until_free <= 0) {
      int least_recently_used_sector = 0;
      for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
        if (env->observations[i] < env->observations[least_recently_used_sector]) {
          least_recently_used_sector = i;
        }
      }
      // X-band searches the least recently used sector
      search_sector_with_range(env, least_recently_used_sector, X_BAND_MAX_RANGE);
      // X-band search covers one sector while S-band search covers a 2x2
      // macro-sector. Keep the fixed surveillance reward area-normalized.
      env->rewards[0] += 0.25f * env->search_action_reward;
      env->search_debt_ms = 0.0f;
      env->x_band_t_until_free = SEARCH_DWELL_TIME;
    }
  }
  else if (action == env->max_trackers + 1 || action == env->max_trackers + 2) {
    // NOOP/IDLE action:
    // consume time without searching/tracking so schedulers can model true idle.
    env->s_band_t_until_free = SEARCH_DWELL_TIME;
    if (env->enable_x_band) {
      env->x_band_t_until_free = SEARCH_DWELL_TIME;
    }
  }
  // else if (action == TRACK1 || action == TRACK2 || action == TRACK3 || action
  // == TRACK4 || action == TRACK5)
  else if (action <= env->max_trackers) {
    bool emitted_track_beam = false;
    float target_range;
    int covered_sector;

    action -= 1;  // easier than -1 all over the place for indexing.

    if (!env->targets[action].is_tracked) {
      env->rewards[0] = -env->track_loss_penalty * (1.0f + 2.0f * env->targets[action].priority);
    } else {
      env->rewards[0] = env->track_update_reward;
      {
        float t_des_before =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER];
        float t_dead_before =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER + 1];
        float priority_scale = 1.0f + 2.0f * env->targets[action].priority;
        float overdue_before = (t_des_before < 0.0f) ? -t_des_before : 0.0f;
        float deadline_pressure =
            (t_dead_before < 100.0f) ? (100.0f - t_dead_before) : 0.0f;
        env->rewards[0] += env->track_urgency_bonus_weight *
                           priority_scale *
                           (overdue_before * GLOBAL_DELAY_PENALTY +
                            0.25f * deadline_pressure * GLOBAL_DELAY_PENALTY);
      }
      // Penalize the delay in updating the tracker
      if (env->enable_local_delay &&
          env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER] < 0) {
          env->rewards[0] += (float)(env->observations[MAX_AZ_SLICES * MAX_EL_SLICES +
                                                       action * FEATURES_PER_TRACKER] *
                                   TRACK_DELAY_PENALTY * (1 + 2 * env->targets[action].priority));
      }

      target_range = get_target_range(env, action);
      bool use_s_band =
          requested_sensor != SENSOR_X_BAND &&
          env->s_band_t_until_free <= 0 &&
          target_range < S_BAND_MAX_RANGE &&
          target_range > S_BAND_MIN_RANGE;
      bool use_x_band =
          requested_sensor != SENSOR_S_BAND &&
          env->enable_x_band &&
          env->x_band_t_until_free <= 0 &&
          target_range < X_BAND_MAX_RANGE &&
          target_range > X_BAND_MIN_RANGE;
      if (use_s_band || use_x_band) {
        update_tracker(env, action);
        emitted_track_beam = true;
      }

      if (emitted_track_beam && env->enable_track_beam_scan) {
        // Track beams also illuminate their own az/el bin. We conservatively
        // search only out to the tracked target range, not beyond it.
        // This pseudo-search is discovery-only: it can reveal hidden targets
        // but must not refresh already tracked targets.
        covered_sector = target_sector(env, action);
        if (covered_sector >= 0) {
          scan_sector_discovery_only(env, covered_sector, target_range);
          env->observations[covered_sector] = ZERO_COST_SEARCH_TIME;
        }
      }

      if (use_s_band) {
        env->s_band_t_until_free =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER + 2];
      } else if (use_x_band) {
        // TODO: actually calculate t_dwell for x-band, for now, just
        // make it faster than s-band
        env->x_band_t_until_free =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + action * FEATURES_PER_TRACKER + 2] /
            2;
      }
    }
  } else {
    // TODO: Throw error?
  }

  // Move simulation forward
move_simulation_forward:
  int delta_t = env->s_band_t_until_free;
  env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] =
      S_BAND_SENSOR;
  if (requested_sensor == SENSOR_X_BAND && env->x_band_t_until_free > 0) {
    delta_t = env->x_band_t_until_free;
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] =
        X_BAND_SENSOR;
  } else if (requested_sensor == SENSOR_S_BAND && env->s_band_t_until_free > 0) {
    delta_t = env->s_band_t_until_free;
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] =
        S_BAND_SENSOR;
  } else if (env->enable_x_band && env->x_band_t_until_free < delta_t) {
    delta_t = env->x_band_t_until_free;
    env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] =
        X_BAND_SENSOR;
  }

  if (delta_t > 0) {
    float global_overdue_increment = 0.0f;
    float sector_stale_before_total = 0.0f;
    float sector_stale_after_total = 0.0f;
    env->tick += delta_t;
    env->search_debt_ms += delta_t;
    env->s_band_t_until_free -= delta_t;
    if (env->enable_x_band) {
      env->x_band_t_until_free -= delta_t;
      if (env->x_band_t_until_free < 0) {
        env->x_band_t_until_free = 0;
      }
    }
    if (env->s_band_t_until_free < 0) {
      env->s_band_t_until_free = 0;
    }
    for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
      float sector_before = env->observations[i];
      env->observations[i] -= delta_t;
      if (env->sector_staleness_weight > 0.0f) {
        float stale_before = (sector_before < 0.0f) ? -sector_before : 0.0f;
        float stale_after = (env->observations[i] < 0.0f) ? -env->observations[i] : 0.0f;
        sector_stale_before_total += stale_before;
        sector_stale_after_total += stale_after;
      }
    }
    if (env->search_frame_overdue_weight > 0.0f && env->search_frame_desired_ms > 0.0f) {
      float frame_cost = 0.0f;
      for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
        float age = ZERO_COST_SEARCH_TIME - env->observations[i];
        float overdue = age - env->search_frame_desired_ms;
        if (overdue > 0.0f) {
          float norm = overdue / env->search_frame_desired_ms;
          frame_cost += norm * norm;
        }
        if (env->search_frame_drop_penalty > 0.0f &&
            env->search_frame_deadline_ms > 0.0f &&
            age > env->search_frame_deadline_ms) {
          frame_cost += env->search_frame_drop_penalty;
        }
      }
      env->rewards[0] -= env->search_frame_overdue_weight *
                         frame_cost /
                         (float)(MAX_AZ_SLICES * MAX_EL_SLICES);
    }
    for (int i = 0; i < env->max_trackers; i++) {
      if (env->targets[i].is_tracked) {
        float t_des_before =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER];
        env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER] -= delta_t;
        env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER + 1] -= delta_t;
        if (env->enable_global_delay) {
          float t_des_after =
              env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER];
          float overdue_before = (t_des_before < 0.0f) ? -t_des_before : 0.0f;
          float overdue_after = (t_des_after < 0.0f) ? -t_des_after : 0.0f;
          float overdue_inc = overdue_after - overdue_before;
          if (overdue_inc > 0.0f) {
              global_overdue_increment += overdue_inc * (1.0f + 2.0f * env->targets[i].priority);
          }
        }
        // if the tracker has expired, lose the track and apply the penalty
        // drop the track if t_deadline < 0
        if (env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER + 1] < 0) {
          // log the loss and all target stats in a csv file
          // FILE *file = fopen("./target_stats.csv", "a");
          // fprintf(file, "%f,%f,%f,%f,%f,%f,%f,%f,%f\n", env->targets[i].x, env->targets[i].y,
          //        env->targets[i].z, env->targets[i].x_velocity, env->targets[i].y_velocity,
          //        env->targets[i].z_velocity, env->targets[i].singer_sigma, env->targets[i].singer_theta,
          //        env->targets[i].priority);
          // fclose(file);
          // printf("Tracker %d lost\n", i);
          // printf("Target range: %f\n", sqrt(env->targets[i].x * env->targets[i].x +
          //                                  env->targets[i].y * env->targets[i].y +
          //                                  env->targets[i].z * env->targets[i].z));
          // printf("Target velocity: %f\n", sqrt(env->targets[i].x_velocity * env->targets[i].x_velocity +
          //                                    env->targets[i].y_velocity * env->targets[i].y_velocity +
          //                                    env->targets[i].z_velocity * env->targets[i].z_velocity));
          // printf("Target singer_sigma: %f\n", env->targets[i].singer_sigma);
          // printf("Target singer_theta: %f\n", env->targets[i].singer_theta);
          // printf("Target priority: %f\n", env->targets[i].priority);
          env->targets[i].is_tracked = false;
          env->rewards[0] -= env->track_loss_penalty * (1.0f + 2.0f * env->targets[i].priority);
          if (env->penalize_hidden_targets) {
            float az_norm = 0.0f;
            float el_norm = 0.0f;
            compute_tracker_timers(env, i,
                                   &env->targets[i].latent_t_desired,
                                   &env->targets[i].latent_t_deadline,
                                   &env->targets[i].latent_t_dwell_estimate,
                                   &az_norm, &el_norm);
          } else {
            env->targets[i].latent_t_desired = NO_TARGET;
            env->targets[i].latent_t_deadline = NO_TARGET;
            env->targets[i].latent_t_dwell_estimate = 0.0f;
          }
        }
        env->targets[i].latent_t_desired =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER];
        env->targets[i].latent_t_deadline =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER + 1];
        env->targets[i].latent_t_dwell_estimate =
            env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + i * FEATURES_PER_TRACKER + 2];
        env->targets[i].service_age_ms += delta_t;
        // t_dwell_estimate does not change
      } else if (env->targets[i].is_active) {
        float t_des_before = env->targets[i].latent_t_desired;
        env->targets[i].latent_t_desired -= delta_t;
        env->targets[i].latent_t_deadline -= delta_t;
        if (env->penalize_hidden_targets && env->enable_global_delay) {
          float overdue_before = (t_des_before < 0.0f) ? -t_des_before : 0.0f;
          float overdue_after =
              (env->targets[i].latent_t_desired < 0.0f) ? -env->targets[i].latent_t_desired : 0.0f;
          float overdue_inc = overdue_after - overdue_before;
          if (overdue_inc > 0.0f) {
            global_overdue_increment += overdue_inc * (1.0f + 2.0f * env->targets[i].priority);
          }
        }
        if (env->penalize_hidden_targets && env->targets[i].latent_t_deadline < 0.0f) {
          env->rewards[0] -= env->track_loss_penalty * (1.0f + 2.0f * env->targets[i].priority);
          float az_norm = 0.0f;
          float el_norm = 0.0f;
          compute_tracker_timers(env, i,
                                 &env->targets[i].latent_t_desired,
                                 &env->targets[i].latent_t_deadline,
                                 &env->targets[i].latent_t_dwell_estimate,
                                 &az_norm, &el_norm);
        }
      }
    }

    // Global delay mode: apply overdue increment per step.
    // This redistributes the same local-delay mass over time
    // (service-time spike -> dense per-step bleed) without changing units.
    if (env->enable_global_delay) {
      env->rewards[0] -= global_overdue_increment * GLOBAL_DELAY_PENALTY;
    }
    if (env->target_service_weight > 0.0f) {
      env->rewards[0] += target_service_before - target_service_cost(env);
    }
    if (env->sector_staleness_weight > 0.0f) {
      env->rewards[0] -= sector_staleness_penalty(
          sector_stale_before_total / (float)(MAX_AZ_SLICES * MAX_EL_SLICES),
          sector_stale_after_total / (float)(MAX_AZ_SLICES * MAX_EL_SLICES),
          env->sector_staleness_weight);
    }
    env->rewards[0] -= search_delay_penalty(
        env->search_debt_ms,
        env->search_delay_mode,
        env->search_debt_penalty_weight,
        env->search_debt_tau_ms,
        env->search_delay_penalty_cap);

    // Update locations

    for (int i = 0; i < env->max_trackers; i++) {
      env->targets[i].x +=
          env->targets[i].x_velocity * delta_t / 1000.0f +
          env->targets[i].x_acceleration * (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].y +=
          env->targets[i].y_velocity * delta_t / 1000.0f +
          env->targets[i].y_acceleration * (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].z +=
          env->targets[i].z_velocity * delta_t / 1000.0f +
          env->targets[i].z_acceleration * (delta_t / 1000.0f) * (delta_t / 1000.0f);
      env->targets[i].x_velocity += env->targets[i].x_acceleration * delta_t / 1000.0f;
      env->targets[i].y_velocity += env->targets[i].y_acceleration * delta_t / 1000.0f;
      env->targets[i].z_velocity += env->targets[i].z_acceleration * delta_t / 1000.0f;
      float rho = exp(-(delta_t / 1000.0f) / env->targets[i].singer_theta);
      env->targets[i].x_acceleration +=
          sqrt(1 - rho * rho) * normal(env, 0, env->targets[i].singer_sigma);
      env->targets[i].y_acceleration +=
          sqrt(1 - rho * rho) * normal(env, 0, env->targets[i].singer_sigma);
      env->targets[i].z_acceleration +=
          sqrt(1 - rho * rho) * normal(env, 0, env->targets[i].singer_sigma) * VERTICAL_MOTION_FACTOR;
    }

    // Reset targets that have gone out of bounds
    for (int i = 0; i < env->max_trackers; i++) {
      if (env->targets[i].x < 0 || env->targets[i].x > MAX_TARGET_XY_RANGE ||
          env->targets[i].y < 0 || env->targets[i].y > MAX_TARGET_XY_RANGE ||
          env->targets[i].z < 0 || env->targets[i].z > MAX_TARGET_Z_RANGE ||
          sqrt(env->targets[i].x * env->targets[i].x + env->targets[i].y * env->targets[i].y +
                  env->targets[i].z * env->targets[i].z) >
              S_BAND_MAX_RANGE) {
        initialize_target(env, i);
      }
    }

    maybe_spawn_poisson_arrivals(env, delta_t);

    // No hard time-limit termination in benchmarking mode.
    (void)env->episode_time_limit_ms;
  }
}

// Required function. Should handle creating the client on first call
void c_render(Radarxs* env) {
#ifndef NO_RAYLIB
// The plan position indicator will be square on left of screen
  // Origin will be at bottom-left
  // X axis positive to right
  // Y axis positive to top
  // I'm going to initially just try to do a WINDOW_Y_PX x WINDOW_Y_PX ppi with
  // a 240 x 80 search indicator on the top right and just hope they don't
  // overlap. int ppi_dimension = WINDOW_Y_PX * scale; int
  // search_indicator_width = 8 * scale;

  int scale = 2;

  if (!IsWindowReady()) {
    if (!(scale == 1 || scale == 2 || scale == 4 || scale == 8)) {
      fprintf(stderr, "Error: scale is one of 1,2,4,8. (4 is 1080p, 8 is 4k).\n");
      exit(1);
    }
    InitWindow(WINDOW_X_PX * scale, WINDOW_Y_PX * scale, "PufferLib Radars");
    SetTargetFPS(10);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(color_bgteal);

  // Draw the search indicator
  int cell_width = 9 * scale;
  int cell_height = 9 * scale;
  int grid_x = WINDOW_X_PX * scale - MAX_AZ_SLICES * cell_width;

  for (int i = 0; i < MAX_EL_SLICES; i++) {
    for (int j = 0; j < MAX_AZ_SLICES; j++) {
      int sector = i * MAX_AZ_SLICES + j;
      int zero_cost_time_remaining = env->observations[sector];
      Color color;
      if (zero_cost_time_remaining >= ZERO_COST_SEARCH_TIME - SEARCH_DWELL_TIME) {
        color = color_red;
      } else if (zero_cost_time_remaining > 0) {
        // Map zero_cost_time_remaining to a cyan - to - grey scale
        float good_intensity = (float)zero_cost_time_remaining / ZERO_COST_SEARCH_TIME;
        if (good_intensity < 0) {
          good_intensity = 0;
        }
        if (good_intensity > 1) {
          good_intensity = 1;
        }
        color = Fade(color_cyan, good_intensity);

      } else if (zero_cost_time_remaining < 0) {
        // Map zero_cost_time_remaining to a yellow - to - red scale
        float red_intensity = (float)abs(zero_cost_time_remaining) / ZERO_COST_SEARCH_TIME;
        if (red_intensity < 0) {
          red_intensity = 0;
        }
        if (red_intensity > 1) {
          red_intensity = 1;
        }
        color = (Color){(unsigned char)(255), (unsigned char)(255 - red_intensity * 255),
                        (unsigned char)(0), 255};
      } else {
        color = GRAY;
      }
      DrawRectangle(grid_x + j * cell_width, i * cell_height, cell_width, cell_height, color);
    }
  }

  // Draw the sensor ranges

  float ppmm = WINDOW_Y_PX * scale / S_BAND_MAX_RANGE;
  Vector2 center = {0, WINDOW_Y_PX * scale};

  // DrawRingLines(center, innerRadius, outerRadius, startAngle, endAngle,
  // (int)segments, color)
  DrawRingLines(center, S_BAND_MIN_RANGE * ppmm, S_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                color_gray);
  if (env->observations[MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER] ==
      S_BAND_SENSOR) {
    DrawRingLines(center, X_BAND_MIN_RANGE * ppmm, X_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                  color_gray);
    DrawRingLines(center, S_BAND_MIN_RANGE * ppmm, S_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                  color_yellow);

  } else {
    DrawRingLines(center, S_BAND_MIN_RANGE * ppmm, S_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                  color_gray);
    DrawRingLines(center, X_BAND_MIN_RANGE * ppmm, X_BAND_MAX_RANGE * ppmm, -90.0f, 0.0f, 0.0f,
                  color_yellow);
  }

  int action = env->actions[0];
  if (action == SEARCH) {
    DrawText("SEARCH", 130 * scale, 5 * scale, 10 * scale, WHITE);
  } else if (action <= env->max_trackers) {
    DrawText(TextFormat("TRACK %d", action), 130 * scale, 5 * scale,
             10 * scale, WHITE);
    action -= 1;
    // if (env->targets[action].is_tracked) {
    //   DrawCircle(env->targets[action].x * ppmm,
    //              WINDOW_Y_PX * scale - env->targets[action].y * ppmm, 10 * scale,
    //              color_red);
    // }
  }

  

  DrawText(TextFormat("TIME: %d ms", env->tick), 130 * scale, 15 * scale,
           10 * scale, WHITE);

  // Print the reward, print the reward per step
  DrawText(TextFormat("REWARD: %.2f", env->rewards[0]), 280 * scale, 100 * scale,
          10 * scale, WHITE);
  DrawText(TextFormat("SUM REWARD: %.2f", env->log.episode_return), 280 * scale, 110 * scale,
           10 * scale, WHITE);
  DrawText(TextFormat("REWARD/SECOND: %.4f", env->log.episode_return / env->tick * 1000), 280 * scale,
           130 * scale, 10 * scale, WHITE);
  
  // The time remaining for each sensor
  DrawText(TextFormat("S-BAND BUSY: %d ms", env->s_band_t_until_free), 280 * scale, 140 * scale,
           10 * scale, WHITE);
  DrawText(TextFormat("X-BAND BUSY: %d ms", env->x_band_t_until_free), 280 * scale, 150 * scale,
           10 * scale, WHITE);

  // Draw the targets
  for (int i = 0; i < env->max_trackers; i++) {
    if (env->targets[i].is_active) {
      float x = env->targets[i].x * ppmm;
      float y = env->targets[i].y * ppmm;
      Vector2 position = {x, WINDOW_Y_PX * scale - y};
      float heading = atan2f(env->targets[i].y_velocity, env->targets[i].x_velocity);
      float arrow_size = 2 * scale;
      if (i == action) {
        arrow_size = 3 * scale;
      }
      Vector2 points[3] = {
          (Vector2){position.x + arrow_size * cosf(heading + PI / 2),
                    position.y + arrow_size * sinf(heading + PI / 2)},
          (Vector2){position.x + 3 * arrow_size * cosf(heading),
                    position.y + 3 * arrow_size * sinf(heading)},
          (Vector2){position.x + arrow_size * cosf(heading - PI / 2),
                    position.y + arrow_size * sinf(heading - PI / 2)},
      };
      if (env->targets[i].is_tracked) {
        if (i == action) {
          DrawTriangle(points[0], points[1], points[2], color_red);
        } else if (env->targets[i].priority >= 2) {
          DrawTriangle(points[0], points[1], points[2], color_blue);
        } else if (env->targets[i].priority >= 1) {
          DrawTriangle(points[0], points[1], points[2], color_sky);
        } else {
          DrawTriangle(points[0], points[1], points[2], color_cyan);
        }
      } else {
        // This draws untracked targets
        // DrawTriangle(points[0], points[1], points[2], color_silver);
      }
    }
  }

  EndDrawing();
#endif
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Radarxs* env) {
#ifndef NO_RAYLIB
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}

