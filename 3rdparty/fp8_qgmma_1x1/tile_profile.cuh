#ifndef TILE_PROFILE_CUH
#define TILE_PROFILE_CUH

struct tile_profile {
    uint64_t scheduler_fetch_start;
    uint64_t scheduler_fetch_complete;
    uint64_t dma_tile_wait_start;
    uint64_t dma_tile_wait_complete;
    uint64_t dma_loads_issued;
    uint64_t compute_tile_wait_start;
    uint64_t compute_tile_wait_complete;
    uint64_t compute_first_data_wait_complete;
    uint64_t epilogue_begin;
    uint64_t epilogue_complete;
    int sm_id;
};

struct ProfileParams {
    int num_tiles;
    int tiles_n;
    tile_profile* profile;
};

#endif // TILE_PROFILE_CUH