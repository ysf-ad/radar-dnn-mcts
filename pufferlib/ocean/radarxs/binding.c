#include <Python.h>
#include <numpy/arrayobject.h>
#include "radarxs.h"

#define Env Radarxs

typedef struct {
    Radarxs** envs;
    int num_envs;
} RadarxsVecEnv;

static Radarxs* radarxs_unpack_vec_first(PyObject* args) {
    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_TypeError, "expected vec env handle");
        return NULL;
    }
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec env handle must be an integer");
        return NULL;
    }
    RadarxsVecEnv* vec = (RadarxsVecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec || vec->num_envs <= 0 || !vec->envs || !vec->envs[0]) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    return vec->envs[0];
}

static PyObject* radarxs_vec_snapshot(PyObject* self, PyObject* args) {
    Radarxs* env = radarxs_unpack_vec_first(args);
    if (!env) {
        return NULL;
    }
    PyObject* dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    int obs_count = MAX_AZ_SLICES * MAX_EL_SLICES + env->max_trackers * FEATURES_PER_TRACKER + PLACEHOLDER_FOR_SENSOR_ID;
    PyDict_SetItemString(dict, "env", PyBytes_FromStringAndSize((const char*)env, sizeof(Radarxs)));
    PyDict_SetItemString(dict, "targets", PyBytes_FromStringAndSize((const char*)env->targets, sizeof(Target) * env->max_trackers));
    PyDict_SetItemString(dict, "observations", PyBytes_FromStringAndSize((const char*)env->observations, sizeof(float) * obs_count));
    PyDict_SetItemString(dict, "action0", PyLong_FromLong(env->actions[0]));
    PyDict_SetItemString(dict, "reward0", PyFloat_FromDouble(env->rewards[0]));
    PyDict_SetItemString(dict, "terminal0", PyLong_FromLong(env->terminals[0]));
    return dict;
}

static PyObject* radarxs_vec_restore(PyObject* self, PyObject* args) {
    Radarxs* env = radarxs_unpack_vec_first(args);
    if (!env) {
        return NULL;
    }
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_restore requires vec handle and snapshot dict");
        return NULL;
    }
    PyObject* dict = PyTuple_GetItem(args, 1);
    if (!PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError, "snapshot must be a dict");
        return NULL;
    }
    char* env_bytes;
    Py_ssize_t env_len;
    char* target_bytes;
    Py_ssize_t target_len;
    char* obs_bytes;
    Py_ssize_t obs_len;
    PyObject* env_obj = PyDict_GetItemString(dict, "env");
    PyObject* target_obj = PyDict_GetItemString(dict, "targets");
    PyObject* obs_obj = PyDict_GetItemString(dict, "observations");
    if (!env_obj || !target_obj || !obs_obj ||
        PyBytes_AsStringAndSize(env_obj, &env_bytes, &env_len) < 0 ||
        PyBytes_AsStringAndSize(target_obj, &target_bytes, &target_len) < 0 ||
        PyBytes_AsStringAndSize(obs_obj, &obs_bytes, &obs_len) < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid snapshot byte payload");
        return NULL;
    }
    if (env_len != sizeof(Radarxs) || target_len != (Py_ssize_t)(sizeof(Target) * env->max_trackers)) {
        PyErr_SetString(PyExc_ValueError, "snapshot shape does not match current env");
        return NULL;
    }
    float* observations = env->observations;
    int* actions = env->actions;
    float* rewards = env->rewards;
    unsigned char* terminals = env->terminals;
    Target* targets = env->targets;
    memcpy(env, env_bytes, sizeof(Radarxs));
    env->observations = observations;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;
    env->targets = targets;
    memcpy(env->targets, target_bytes, (size_t)target_len);
    memcpy(env->observations, obs_bytes, (size_t)obs_len);
    PyObject* action_obj = PyDict_GetItemString(dict, "action0");
    PyObject* reward_obj = PyDict_GetItemString(dict, "reward0");
    PyObject* terminal_obj = PyDict_GetItemString(dict, "terminal0");
    if (action_obj) env->actions[0] = (int)PyLong_AsLong(action_obj);
    if (reward_obj) env->rewards[0] = (float)PyFloat_AsDouble(reward_obj);
    if (terminal_obj) env->terminals[0] = (unsigned char)PyLong_AsLong(terminal_obj);
    Py_RETURN_NONE;
}

static int radarxs_restore_snapshot_into(Radarxs* env, PyObject* dict) {
    char* env_bytes;
    Py_ssize_t env_len;
    char* target_bytes;
    Py_ssize_t target_len;
    char* obs_bytes;
    Py_ssize_t obs_len;
    PyObject* env_obj = PyDict_GetItemString(dict, "env");
    PyObject* target_obj = PyDict_GetItemString(dict, "targets");
    PyObject* obs_obj = PyDict_GetItemString(dict, "observations");
    if (!env_obj || !target_obj || !obs_obj ||
        PyBytes_AsStringAndSize(env_obj, &env_bytes, &env_len) < 0 ||
        PyBytes_AsStringAndSize(target_obj, &target_bytes, &target_len) < 0 ||
        PyBytes_AsStringAndSize(obs_obj, &obs_bytes, &obs_len) < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid snapshot byte payload");
        return -1;
    }
    if (env_len != sizeof(Radarxs) || target_len != (Py_ssize_t)(sizeof(Target) * env->max_trackers)) {
        PyErr_SetString(PyExc_ValueError, "snapshot shape does not match current env");
        return -1;
    }
    float* observations = env->observations;
    int* actions = env->actions;
    float* rewards = env->rewards;
    unsigned char* terminals = env->terminals;
    Target* targets = env->targets;
    memcpy(env, env_bytes, sizeof(Radarxs));
    env->observations = observations;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;
    env->targets = targets;
    memcpy(env->targets, target_bytes, (size_t)target_len);
    memcpy(env->observations, obs_bytes, (size_t)obs_len);
    PyObject* action_obj = PyDict_GetItemString(dict, "action0");
    PyObject* reward_obj = PyDict_GetItemString(dict, "reward0");
    PyObject* terminal_obj = PyDict_GetItemString(dict, "terminal0");
    if (action_obj) env->actions[0] = (int)PyLong_AsLong(action_obj);
    if (reward_obj) env->rewards[0] = (float)PyFloat_AsDouble(reward_obj);
    if (terminal_obj) env->terminals[0] = (unsigned char)PyLong_AsLong(terminal_obj);
    return 0;
}

static PyObject* radarxs_vec_restore_all(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_restore_all requires vec handle and snapshot dict");
        return NULL;
    }
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec env handle must be an integer");
        return NULL;
    }
    RadarxsVecEnv* vec = (RadarxsVecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec || vec->num_envs <= 0 || !vec->envs) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    PyObject* dict = PyTuple_GetItem(args, 1);
    if (!PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError, "snapshot must be a dict");
        return NULL;
    }
    for (int i = 0; i < vec->num_envs; i++) {
        if (!vec->envs[i]) {
            PyErr_SetString(PyExc_ValueError, "invalid env in vector");
            return NULL;
        }
        if (radarxs_restore_snapshot_into(vec->envs[i], dict) < 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* radarxs_vec_restore_n(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_restore_n requires vec handle, snapshot dict, and count");
        return NULL;
    }
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec env handle must be an integer");
        return NULL;
    }
    RadarxsVecEnv* vec = (RadarxsVecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec || vec->num_envs <= 0 || !vec->envs) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    PyObject* dict = PyTuple_GetItem(args, 1);
    if (!PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError, "snapshot must be a dict");
        return NULL;
    }
    int count = (int)PyLong_AsLong(PyTuple_GetItem(args, 2));
    if (count < 0 || count > vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "restore count out of range");
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        if (!vec->envs[i]) {
            PyErr_SetString(PyExc_ValueError, "invalid env in vector");
            return NULL;
        }
        if (radarxs_restore_snapshot_into(vec->envs[i], dict) < 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* radarxs_vec_restore_many(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_restore_many requires vec handle, snapshot list, and count");
        return NULL;
    }
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec env handle must be an integer");
        return NULL;
    }
    RadarxsVecEnv* vec = (RadarxsVecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec || vec->num_envs <= 0 || !vec->envs) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    PyObject* snapshots_obj = PyTuple_GetItem(args, 1);
    PyObject* snapshots = PySequence_Fast(snapshots_obj, "snapshots must be a sequence of snapshot dicts");
    if (!snapshots) {
        PyErr_SetString(PyExc_TypeError, "snapshots must be a sequence of snapshot dicts");
        return NULL;
    }
    int count = (int)PyLong_AsLong(PyTuple_GetItem(args, 2));
    if (count < 0 || count > vec->num_envs || PySequence_Fast_GET_SIZE(snapshots) < count) {
        Py_DECREF(snapshots);
        PyErr_SetString(PyExc_ValueError, "restore count out of range");
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(snapshots, i);
        if (!item || !PyDict_Check(item)) {
            Py_DECREF(snapshots);
            PyErr_SetString(PyExc_TypeError, "snapshot item must be a dict");
            return NULL;
        }
        if (!vec->envs[i]) {
            Py_DECREF(snapshots);
            PyErr_SetString(PyExc_ValueError, "invalid env in vector");
            return NULL;
        }
        if (radarxs_restore_snapshot_into(vec->envs[i], item) < 0) {
            Py_DECREF(snapshots);
            return NULL;
        }
    }
    Py_DECREF(snapshots);
    Py_RETURN_NONE;
}

static PyObject* radarxs_vec_aux(PyObject* self, PyObject* args) {
    Radarxs* env = radarxs_unpack_vec_first(args);
    if (!env) {
        return NULL;
    }
    PyObject* dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    PyObject* ranges = PyList_New(env->max_trackers);
    if (!ranges) {
        Py_DECREF(dict);
        return NULL;
    }
    for (int i = 0; i < env->max_trackers; i++) {
        float r = sqrtf(env->targets[i].x * env->targets[i].x +
                        env->targets[i].y * env->targets[i].y +
                        env->targets[i].z * env->targets[i].z);
        PyList_SET_ITEM(ranges, i, PyFloat_FromDouble((double)r));
    }
    PyDict_SetItemString(dict, "s_band_busy_ms", PyLong_FromLong(env->s_band_t_until_free));
    PyDict_SetItemString(dict, "x_band_busy_ms", PyLong_FromLong(env->x_band_t_until_free));
    PyDict_SetItemString(dict, "enable_x_band", PyLong_FromLong(env->enable_x_band));
    PyDict_SetItemString(dict, "target_range", ranges);
    Py_DECREF(ranges);
    return dict;
}

static PyObject* radarxs_aux_one(Radarxs* env) {
    PyObject* dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    PyObject* ranges = PyList_New(env->max_trackers);
    if (!ranges) {
        Py_DECREF(dict);
        return NULL;
    }
    for (int i = 0; i < env->max_trackers; i++) {
        float r = sqrtf(env->targets[i].x * env->targets[i].x +
                        env->targets[i].y * env->targets[i].y +
                        env->targets[i].z * env->targets[i].z);
        PyList_SET_ITEM(ranges, i, PyFloat_FromDouble((double)r));
    }
    PyDict_SetItemString(dict, "s_band_busy_ms", PyLong_FromLong(env->s_band_t_until_free));
    PyDict_SetItemString(dict, "x_band_busy_ms", PyLong_FromLong(env->x_band_t_until_free));
    PyDict_SetItemString(dict, "enable_x_band", PyLong_FromLong(env->enable_x_band));
    PyDict_SetItemString(dict, "target_range", ranges);
    Py_DECREF(ranges);
    return dict;
}

static PyObject* radarxs_vec_aux_all(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_TypeError, "expected vec env handle");
        return NULL;
    }
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec env handle must be an integer");
        return NULL;
    }
    RadarxsVecEnv* vec = (RadarxsVecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec || vec->num_envs <= 0 || !vec->envs) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    PyObject* list = PyList_New(vec->num_envs);
    if (!list) {
        return NULL;
    }
    for (int i = 0; i < vec->num_envs; i++) {
        PyObject* item = radarxs_aux_one(vec->envs[i]);
        if (!item) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}

static int radarxs_decode_physical_action(Radarxs* env, int raw_action, int* logical_action, int* requested_sensor) {
    int s_search_action = env->max_trackers + 3;
    int x_search_action = env->max_trackers + 4;
    int s_track_base = env->max_trackers + 5;
    int x_track_base = env->max_trackers + 5 + env->max_trackers;
    *logical_action = raw_action;
    *requested_sensor = SENSOR_IMPLICIT;
    if (raw_action == s_search_action) {
        *logical_action = SEARCH;
        *requested_sensor = SENSOR_S_BAND;
    } else if (raw_action == x_search_action) {
        *logical_action = SEARCH;
        *requested_sensor = SENSOR_X_BAND;
    } else if (raw_action >= s_track_base && raw_action < s_track_base + env->max_trackers) {
        *logical_action = (raw_action - s_track_base) + 1;
        *requested_sensor = SENSOR_S_BAND;
    } else if (raw_action >= x_track_base && raw_action < x_track_base + env->max_trackers) {
        *logical_action = (raw_action - x_track_base) + 1;
        *requested_sensor = SENSOR_X_BAND;
    }
    return 0;
}

static int radarxs_action_valid_for_wrapper(Radarxs* env, int raw_action) {
    int logical_action;
    int requested_sensor;
    radarxs_decode_physical_action(env, raw_action, &logical_action, &requested_sensor);
    if (env->terminals[0]) {
        return 0;
    }
    if (logical_action == SEARCH) {
        if (requested_sensor == SENSOR_S_BAND && env->s_band_t_until_free > 0) {
            return 0;
        }
        if (requested_sensor == SENSOR_X_BAND &&
            (!env->enable_x_band || env->x_band_t_until_free > 0)) {
            return 0;
        }
        return 1;
    }
    if (logical_action > 0) {
        int idx = logical_action - 1;
        if (idx < 0 || idx >= env->max_trackers) {
            return 0;
        }
        if (!env->targets[idx].is_active || !env->targets[idx].is_tracked) {
            return 0;
        }
        int base = MAX_AZ_SLICES * MAX_EL_SLICES + idx * FEATURES_PER_TRACKER;
        if (env->observations[base + 1] < 0.0f) {
            return 0;
        }
        float range = get_target_range(env, idx);
        if (requested_sensor == SENSOR_S_BAND &&
            !(env->s_band_t_until_free == 0 &&
              range > S_BAND_MIN_RANGE && range < S_BAND_MAX_RANGE)) {
            return 0;
        }
        if (requested_sensor == SENSOR_X_BAND &&
            !(env->enable_x_band &&
              env->x_band_t_until_free == 0 &&
              range > X_BAND_MIN_RANGE && range < X_BAND_MAX_RANGE)) {
            return 0;
        }
        return 1;
    }
    return 1;
}

static PyObject* radarxs_vec_step_validated(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_step_validated requires 1 argument");
        return NULL;
    }
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec env handle must be an integer");
        return NULL;
    }
    RadarxsVecEnv* vec = (RadarxsVecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec || vec->num_envs <= 0 || !vec->envs) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    PyObject* dt_list = PyList_New(vec->num_envs);
    PyObject* executed_list = PyList_New(vec->num_envs);
    if (!dt_list || !executed_list) {
        Py_XDECREF(dt_list);
        Py_XDECREF(executed_list);
        return NULL;
    }
    for (int i = 0; i < vec->num_envs; i++) {
        Radarxs* env = vec->envs[i];
        int raw_action = env->actions[0];
        int logical_action;
        int requested_sensor;
        int before_tick = env->tick;
        radarxs_decode_physical_action(env, raw_action, &logical_action, &requested_sensor);
        int valid = radarxs_action_valid_for_wrapper(env, raw_action);
        if (valid) {
            c_step(env);
            double dt = (double)(env->tick - before_tick);
            if (logical_action == SEARCH && dt <= 0.0) {
                dt = (double)SEARCH_DWELL_TIME;
            }
            PyList_SET_ITEM(executed_list, i, PyLong_FromLong(raw_action));
            PyList_SET_ITEM(dt_list, i, PyFloat_FromDouble(dt));
        } else {
            env->rewards[0] = 0.0f;
            PyList_SET_ITEM(dt_list, i, PyFloat_FromDouble(0.0));
            PyList_SET_ITEM(executed_list, i, PyLong_FromLong(-1));
        }
    }
    PyObject* dict = PyDict_New();
    if (!dict) {
        Py_DECREF(dt_list);
        Py_DECREF(executed_list);
        return NULL;
    }
    PyDict_SetItemString(dict, "dt", dt_list);
    PyDict_SetItemString(dict, "executed", executed_list);
    Py_DECREF(dt_list);
    Py_DECREF(executed_list);
    return dict;
}

static PyObject* radarxs_vec_step_validated_into(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 4) {
        PyErr_SetString(PyExc_TypeError, "vec_step_validated_into requires vec handle, dt array, executed array, and count");
        return NULL;
    }
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec env handle must be an integer");
        return NULL;
    }
    RadarxsVecEnv* vec = (RadarxsVecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec || vec->num_envs <= 0 || !vec->envs) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    PyObject* dt_obj = PyTuple_GetItem(args, 1);
    PyObject* executed_obj = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(dt_obj, &PyArray_Type) || !PyObject_TypeCheck(executed_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "dt and executed outputs must be NumPy arrays");
        return NULL;
    }
    PyArrayObject* dt_arr = (PyArrayObject*)dt_obj;
    PyArrayObject* executed_arr = (PyArrayObject*)executed_obj;
    if (!PyArray_ISCONTIGUOUS(dt_arr) || !PyArray_ISCONTIGUOUS(executed_arr)) {
        PyErr_SetString(PyExc_ValueError, "dt and executed outputs must be contiguous");
        return NULL;
    }
    if (PyArray_TYPE(dt_arr) != NPY_FLOAT32 || PyArray_TYPE(executed_arr) != NPY_INT32) {
        PyErr_SetString(PyExc_ValueError, "dt must be float32 and executed must be int32");
        return NULL;
    }
    int count = (int)PyLong_AsLong(PyTuple_GetItem(args, 3));
    if (count < 0 || count > vec->num_envs ||
        PyArray_SIZE(dt_arr) < count || PyArray_SIZE(executed_arr) < count) {
        PyErr_SetString(PyExc_ValueError, "step count out of range");
        return NULL;
    }
    float* dt_out = (float*)PyArray_DATA(dt_arr);
    int* executed_out = (int*)PyArray_DATA(executed_arr);
    for (int i = 0; i < count; i++) {
        Radarxs* env = vec->envs[i];
        int raw_action = env->actions[0];
        int logical_action;
        int requested_sensor;
        int before_tick = env->tick;
        radarxs_decode_physical_action(env, raw_action, &logical_action, &requested_sensor);
        int valid = radarxs_action_valid_for_wrapper(env, raw_action);
        if (valid) {
            c_step(env);
            float dt = (float)(env->tick - before_tick);
            if (logical_action == SEARCH && dt <= 0.0f) {
                dt = (float)SEARCH_DWELL_TIME;
            }
            executed_out[i] = raw_action;
            dt_out[i] = dt;
        } else {
            env->rewards[0] = 0.0f;
            executed_out[i] = -1;
            dt_out[i] = 0.0f;
        }
    }
    return PyLong_FromLong(count);
}

#define MY_METHODS \
    {"vec_snapshot", radarxs_vec_snapshot, METH_VARARGS, "Snapshot first radar env in a vector"}, \
    {"vec_restore", radarxs_vec_restore, METH_VARARGS, "Restore first radar env in a vector"}, \
    {"vec_restore_all", radarxs_vec_restore_all, METH_VARARGS, "Restore every radar env in a vector from one snapshot"}, \
    {"vec_restore_n", radarxs_vec_restore_n, METH_VARARGS, "Restore the first N radar envs in a vector from one snapshot"}, \
    {"vec_restore_many", radarxs_vec_restore_many, METH_VARARGS, "Restore first N radar envs from N snapshot dicts"}, \
    {"vec_aux", radarxs_vec_aux, METH_VARARGS, "Return sensor busy timers and target ranges for first radar env"}, \
    {"vec_aux_all", radarxs_vec_aux_all, METH_VARARGS, "Return sensor busy timers and target ranges for every radar env"}, \
    {"vec_step_validated", radarxs_vec_step_validated, METH_VARARGS, "Step every env with Python-wrapper-compatible action validity"}, \
    {"vec_step_validated_into", radarxs_vec_step_validated_into, METH_VARARGS, "Step first N envs and write dt/executed into NumPy arrays"}

#include "../env_binding.h"

static int my_log(PyObject* dict, Log* log) {
    return 0;  // No custom logging
}

static int optional_int_kwarg(PyObject* kwargs, const char* key, int default_value) {
    PyObject* val;
    if (kwargs == NULL) {
        return default_value;
    }
    val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        return default_value;
    }
    if (PyLong_Check(val)) {
        return (int)PyLong_AsLong(val);
    }
    if (PyFloat_Check(val)) {
        return (int)PyFloat_AsDouble(val);
    }
    return default_value;
}

static float optional_float_kwarg(PyObject* kwargs, const char* key, float default_value) {
    PyObject* val;
    if (kwargs == NULL) {
        return default_value;
    }
    val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        return default_value;
    }
    if (PyFloat_Check(val)) {
        return (float)PyFloat_AsDouble(val);
    }
    if (PyLong_Check(val)) {
        return (float)PyLong_AsLong(val);
    }
    return default_value;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->initial_targets = (int)unpack(kwargs, "initial_targets");
    env->max_trackers = (int)unpack(kwargs, "max_trackers");
    if (env->initial_targets == 0) env->initial_targets = 50; // Default
    if (env->max_trackers == 0) env->max_trackers = 100; // Default
    env->enable_global_delay = optional_int_kwarg(kwargs, "enable_global_delay", 1);
    env->enable_local_delay = optional_int_kwarg(kwargs, "enable_local_delay", 1);
    env->enable_x_band = optional_int_kwarg(kwargs, "enable_x_band", 0);
    env->enable_search_refresh_tracked = optional_int_kwarg(kwargs, "enable_search_refresh_tracked", 0);
    env->search_refresh_gain = optional_float_kwarg(kwargs, "search_refresh_gain", 1.0f);
    env->enable_priority = optional_int_kwarg(kwargs, "enable_priority", 1);
    env->enable_poisson_arrivals = optional_int_kwarg(kwargs, "enable_poisson_arrivals", 0);
    env->activate_all_targets_without_poisson =
        optional_int_kwarg(kwargs, "activate_all_targets_without_poisson", 1);
    env->poisson_rate_per_second = optional_float_kwarg(kwargs, "poisson_rate_per_second", 5.0f);
    env->search_action_reward = optional_float_kwarg(kwargs, "search_action_reward", SEARCH_ACTION_REWARD);
    env->track_update_reward = optional_float_kwarg(kwargs, "track_update_reward", TRACK_UPDATE_REWARD);
    env->track_loss_penalty = optional_float_kwarg(kwargs, "track_loss_penalty", TRACK_LOSS_PENALTY);
    env->track_urgency_bonus_weight = optional_float_kwarg(kwargs, "track_urgency_bonus_weight", 0.0f);
    env->target_service_weight = optional_float_kwarg(kwargs, "target_service_weight", 0.0f);
    env->target_service_horizon_ms = optional_float_kwarg(kwargs, "target_service_horizon_ms", 1000.0f);
    env->sector_staleness_weight = optional_float_kwarg(kwargs, "sector_staleness_weight", 0.0f);
    env->searched_sector_reward_weight = optional_float_kwarg(kwargs, "searched_sector_reward_weight", 0.0f);
    env->search_frame_overdue_weight = optional_float_kwarg(kwargs, "search_frame_overdue_weight", 0.0f);
    env->search_frame_desired_ms = optional_float_kwarg(kwargs, "search_frame_desired_ms", ZERO_COST_SEARCH_TIME);
    env->search_frame_deadline_ms = optional_float_kwarg(kwargs, "search_frame_deadline_ms", 1.5f * ZERO_COST_SEARCH_TIME);
    env->search_frame_drop_penalty = optional_float_kwarg(kwargs, "search_frame_drop_penalty", 0.0f);
    env->search_task_cost_mode = optional_int_kwarg(kwargs, "search_task_cost_mode", 0);
    env->revisit_time_scale = optional_float_kwarg(kwargs, "revisit_time_scale", 1.0f);
    env->dwell_time_scale = optional_float_kwarg(kwargs, "dwell_time_scale", 1.0f);
    env->penalize_hidden_targets = optional_int_kwarg(kwargs, "penalize_hidden_targets", 0);
    env->enable_track_beam_scan = optional_int_kwarg(kwargs, "enable_track_beam_scan", 0);
    // Default effectively "no episode timeout" for long-horizon benchmarking.
    env->episode_time_limit_ms = optional_int_kwarg(kwargs, "episode_time_limit_ms", 2000000000);
    env->search_delay_mode = optional_int_kwarg(kwargs, "search_delay_mode", SEARCH_DELAY_MODE);
    env->search_debt_penalty_weight = optional_float_kwarg(kwargs, "search_debt_penalty_weight", SEARCH_DEBT_PENALTY_WEIGHT);
    env->search_debt_tau_ms = optional_float_kwarg(kwargs, "search_debt_tau_ms", SEARCH_DEBT_TAU_MS);
    env->search_delay_penalty_cap = optional_float_kwarg(
        kwargs, "search_delay_penalty_cap", SEARCH_DELAY_PENALTY_CAP);

    env->targets = (Target*)calloc(env->max_trackers, sizeof(Target));
    return 0;
}
