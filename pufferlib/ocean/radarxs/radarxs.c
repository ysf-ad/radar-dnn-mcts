/* Build it with:
 * bash scripts/build_ocean.sh radarxs local (debug)
 * bash scripts/build_ocean.sh radarxs fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. 
 * Run it on mac with:
 * puffer train puffer_radarxs --train.device mps
 */

#include "radarxs.h"

int main() {
    Radarxs env = {.max_trackers = 500, .initial_targets = 300};
    env.observations = (int16_t*)calloc(MAX_AZ_SLICES * MAX_EL_SLICES + env.max_trackers * FEATURES_PER_TRACKER + PLACEHOLDER_FOR_SENSOR_ID, sizeof(int16_t));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.targets = (Target *)calloc(env.max_trackers, sizeof(Target));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        env.actions[0] = rand() % 5;
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.targets);
    c_close(&env);
}

