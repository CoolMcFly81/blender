/*
 * Copyright 2011-2013 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tile.h"

#include "util_algorithm.h"
#include "util_types.h"

CCL_NAMESPACE_BEGIN

namespace {

class TileComparator {
public:
	TileComparator(TileOrder order_, int2 center_, Tile *tiles_)
	 :  order(order_),
	    center(center_),
	    tiles(tiles_)
	{}

	bool operator()(int a, int b)
	{
		switch(order) {
			case TILE_CENTER:
			{
				float2 dist_a = make_float2(center.x - (tiles[a].x + tiles[a].w/2),
				                            center.y - (tiles[a].y + tiles[a].h/2));
				float2 dist_b = make_float2(center.x - (tiles[b].x + tiles[b].w/2),
				                            center.y - (tiles[b].y + tiles[b].h/2));
				return dot(dist_a, dist_a) < dot(dist_b, dist_b);
			}
			case TILE_LEFT_TO_RIGHT:
				return (tiles[a].x == tiles[b].x)? (tiles[a].y < tiles[b].y): (tiles[a].x < tiles[b].x);
			case TILE_RIGHT_TO_LEFT:
				return (tiles[a].x == tiles[b].x)? (tiles[a].y < tiles[b].y): (tiles[a].x > tiles[b].x);
			case TILE_TOP_TO_BOTTOM:
				return (tiles[a].y == tiles[b].y)? (tiles[a].x < tiles[b].x): (tiles[a].y > tiles[b].y);
			case TILE_BOTTOM_TO_TOP:
			default:
				return (tiles[a].y == tiles[b].y)? (tiles[a].x < tiles[b].x): (tiles[a].y < tiles[b].y);
		}
	}

protected:
	TileOrder order;
	int2 center;
	Tile *tiles;
};

inline int2 hilbert_index_to_pos(int n, int d)
{
	int2 r, xy = make_int2(0, 0);
	for(int s = 1; s < n; s *= 2) {
		r.x = (d >> 1) & 1;
		r.y = (d ^ r.x) & 1;
		if(!r.y) {
			if(r.x) {
				xy = make_int2(s-1, s-1) - xy;
			}
			swap(xy.x, xy.y);
		}
		xy += r*make_int2(s, s);
		d >>= 2;
	}
	return xy;
}

enum SpiralDirection {
	DIRECTION_UP,
	DIRECTION_LEFT,
	DIRECTION_DOWN,
	DIRECTION_RIGHT,
};

}  /* namespace */

TileManager::TileManager(bool progressive_, int num_samples_, int2 tile_size_, int start_resolution_,
                         bool preserve_tile_device_, bool background_, TileOrder tile_order_, int num_devices_, bool only_denoise_)
{
	progressive = progressive_;
	tile_size = tile_size_;
	tile_order = tile_order_;
	start_resolution = start_resolution_;
	num_samples = num_samples_;
	num_devices = num_devices_;
	preserve_tile_device = preserve_tile_device_;
	background = background_;
	schedule_denoising = false;
	only_denoise = only_denoise_;

	range_start_sample = 0;
	range_num_samples = -1;

	BufferParams buffer_params;
	reset(buffer_params, 0);
}

TileManager::~TileManager()
{
}

void TileManager::free_device()
{
	if(schedule_denoising) {
		for(int i = 0; i < state.tiles.size(); i++) {
			delete state.tiles[i].buffers;
			state.tiles[i].buffers = NULL;
		}
	}
}

static int get_divider(int w, int h, int start_resolution)
{
	int divider = 1;
	if(start_resolution != INT_MAX) {
		while(w*h > start_resolution*start_resolution) {
			w = max(1, w/2);
			h = max(1, h/2);

			divider <<= 1;
		}
	}
	return divider;
}

void TileManager::reset(BufferParams& params_, int num_samples_)
{
	params = params_;

	set_samples(num_samples_);

	state.buffer = BufferParams();
	state.global_buffers = NULL;
	state.sample = range_start_sample - 1;
	state.num_tiles = 0;
	state.num_rendered_tiles = 0;
	state.num_samples = 0;
	state.resolution_divider = get_divider(params.width, params.height, start_resolution);
	state.render_tiles.clear();
	state.denoise_tiles.clear();
	state.tiles.clear();
}

void TileManager::set_samples(int num_samples_)
{
	num_samples = num_samples_;

	/* No real progress indication is possible when using unlimited samples. */
	if(num_samples == INT_MAX) {
		state.total_pixel_samples = 0;
	}
	else if(only_denoise) {
		state.total_pixel_samples = params.width*params.height;
	}
	else {
		uint64_t pixel_samples = 0;
		/* While rendering in the viewport, the initial preview resolution is increased to the native resolution
		 * before the actual rendering begins. Therefore, additional pixel samples will be rendered. */
		int divider = get_divider(params.width, params.height, start_resolution) / 2;
		while(divider > 1) {
			int image_w = max(1, params.width/divider);
			int image_h = max(1, params.height/divider);
			pixel_samples += image_w * image_h;
			divider >>= 1;
		}

		state.total_pixel_samples = pixel_samples + get_num_effective_samples() * params.width*params.height;
		if(schedule_denoising) {
			state.total_pixel_samples += params.width*params.height;
		}
	}
}

/* If sliced is false, splits image into tiles and assigns equal amount of tiles to every render device.
 * If sliced is true, slice image into as much pieces as how many devices are rendering this image. */
int TileManager::gen_tiles(bool sliced)
{
	int resolution = state.resolution_divider;
	int image_w = max(1, params.width/resolution);
	int image_h = max(1, params.height/resolution);
	int2 center = make_int2(image_w/2, image_h/2);

	int num_logical_devices = preserve_tile_device? num_devices: 1;
	int num = min(image_h, num_logical_devices);
	int slice_num = sliced? num: 1;

	int tile_w = (tile_size.x >= image_w)? 1: (image_w + tile_size.x - 1)/tile_size.x;
	int tile_h = (tile_size.y >= image_h)? 1: (image_h + tile_size.y - 1)/tile_size.y;

	state.tiles.clear();
	state.tiles.resize(tile_w*tile_h);
	state.render_tiles.clear();
	state.denoise_tiles.clear();
	state.render_tiles.resize(num);
	state.denoise_tiles.resize(num);
	state.tile_stride = tile_w;
	vector<list<int> >::iterator tile_list;
	if(only_denoise)
		tile_list = state.denoise_tiles.begin();
	else
		tile_list = state.render_tiles.begin();

	if(tile_order == TILE_HILBERT_SPIRAL) {
		assert(!sliced);

		/* Size of blocks in tiles, must be a power of 2 */
		const int hilbert_size = (max(tile_size.x, tile_size.y) <= 12)? 8: 4;

		int tiles_per_device = (tile_w * tile_h + num - 1) / num;
		int cur_device = 0, cur_tiles = 0;

		int2 block_size = tile_size * make_int2(hilbert_size, hilbert_size);
		/* Number of blocks to fill the image */
		int blocks_x = (block_size.x >= image_w)? 1: (image_w + block_size.x - 1)/block_size.x;
		int blocks_y = (block_size.y >= image_h)? 1: (image_h + block_size.y - 1)/block_size.y;
		int n = max(blocks_x, blocks_y) | 0x1; /* Side length of the spiral (must be odd) */
		/* Offset of spiral (to keep it centered) */
		int2 offset = make_int2((image_w - n*block_size.x)/2, (image_h - n*block_size.y)/2);
		offset = (offset / tile_size) * tile_size; /* Round to tile border. */

		int2 block = make_int2(0, 0); /* Current block */
		SpiralDirection prev_dir = DIRECTION_UP, dir = DIRECTION_UP;
		for(int i = 0;;) {
			/* Generate the tiles in the current block. */
			for(int hilbert_index = 0; hilbert_index < hilbert_size*hilbert_size; hilbert_index++) {
				int2 tile, hilbert_pos = hilbert_index_to_pos(hilbert_size, hilbert_index);
				/* Rotate block according to spiral direction. */
				if(prev_dir == DIRECTION_UP && dir == DIRECTION_UP) {
					tile = make_int2(hilbert_pos.y, hilbert_pos.x);
				}
				else if(dir == DIRECTION_LEFT || prev_dir == DIRECTION_LEFT) {
					tile = hilbert_pos;
				}
				else if(dir == DIRECTION_DOWN) {
					tile = make_int2(hilbert_size-1-hilbert_pos.y, hilbert_size-1-hilbert_pos.x);
				}
				else {
					tile = make_int2(hilbert_size-1-hilbert_pos.x, hilbert_size-1-hilbert_pos.y);
				}

				int2 pos = block*block_size + tile*tile_size + offset;
				/* Only add tiles which are in the image (tiles outside of the image can be generated since the spiral is always square). */
				if(pos.x >= 0 && pos.y >= 0 && pos.x < image_w && pos.y < image_h) {
					int w = min(tile_size.x, image_w - pos.x);
					int h = min(tile_size.y, image_h - pos.y);
					int2 ipos = pos / tile_size;
					int idx = ipos.y*tile_w + ipos.x;
					state.tiles[idx] = Tile(idx, pos.x, pos.y, w, h, cur_device, only_denoise? Tile::DENOISE : Tile::RENDER, state.global_buffers);
					tile_list->push_front(idx);
					cur_tiles++;

					if(cur_tiles == tiles_per_device) {
						tile_list++;
						cur_tiles = 0;
						cur_device++;
					}
				}
			}

			/* Stop as soon as the spiral has reached the center block. */
			if(block.x == (n-1)/2 && block.y == (n-1)/2)
				break;

			/* Advance to next block. */
			prev_dir = dir;
			switch(dir) {
				case DIRECTION_UP:
					block.y++;
					if(block.y == (n-i-1)) {
						dir = DIRECTION_LEFT;
					}
					break;
				case DIRECTION_LEFT:
					block.x++;
					if(block.x == (n-i-1)) {
						dir = DIRECTION_DOWN;
					}
					break;
				case DIRECTION_DOWN:
					block.y--;
					if(block.y == i) {
						dir = DIRECTION_RIGHT;
					}
					break;
				case DIRECTION_RIGHT:
					block.x--;
					if(block.x == i+1) {
						dir = DIRECTION_UP;
						i++;
					}
					break;
			}
		}
		return tile_w*tile_h;
	}

	for(int slice = 0; slice < slice_num; slice++) {
		int slice_y = (image_h/slice_num)*slice;
		int slice_h = (slice == slice_num-1)? image_h - slice*(image_h/slice_num): image_h/slice_num;

		int tile_slice_h = (tile_size.y >= slice_h)? 1: (slice_h + tile_size.y - 1)/tile_size.y;

		int tiles_per_device = (tile_w * tile_slice_h + num - 1) / num;
		int cur_device = 0, cur_tiles = 0;

		for(int tile_y = 0; tile_y < tile_slice_h; tile_y++) {
			for(int tile_x = 0; tile_x < tile_w; tile_x++) {
				int x = tile_x * tile_size.x;
				int y = tile_y * tile_size.y;
				int w = (tile_x == tile_w-1)? image_w - x: tile_size.x;
				int h = (tile_y == tile_slice_h-1)? slice_h - y: tile_size.y;

				int idx = tile_y*tile_w + tile_x;
				state.tiles[idx] = Tile(idx, x, y + slice_y, w, h, sliced? slice: cur_device, only_denoise? Tile::DENOISE : Tile::RENDER, state.global_buffers);
				tile_list->push_back(idx);

				if(!sliced) {
					cur_tiles++;

					if(cur_tiles == tiles_per_device) {
						/* Tiles are already generated in Bottom-to-Top order, so no sort is necessary in that case. */
						if(tile_order != TILE_BOTTOM_TO_TOP) {
							tile_list->sort(TileComparator(tile_order, center, &state.tiles[0]));
						}
						tile_list++;
						cur_tiles = 0;
						cur_device++;
					}
				}
			}
		}
		if(sliced) {
			tile_list++;
		}
	}

	return tile_w*tile_h;
}

void TileManager::set_tiles()
{
	int resolution = state.resolution_divider;
	int image_w = max(1, params.width/resolution);
	int image_h = max(1, params.height/resolution);

	state.num_tiles = gen_tiles(!background);

	state.buffer.width = image_w;
	state.buffer.height = image_h;

	state.buffer.full_x = params.full_x/resolution;
	state.buffer.full_y = params.full_y/resolution;
	state.buffer.full_width = max(1, params.full_width/resolution);
	state.buffer.full_height = max(1, params.full_height/resolution);
}

/* Returns whether the tile should be written (and freed if no denoising is used) instead of updating. */
bool TileManager::return_tile(int index, bool &delete_tile)
{
	int resolution = state.resolution_divider;
	int image_w = max(1, params.width/resolution);
	int image_h = max(1, params.height/resolution);
	int tile_w = (tile_size.x >= image_w)? 1: (image_w + tile_size.x - 1)/tile_size.x;
	int tile_h = (tile_size.y >= image_h)? 1: (image_h + tile_size.y - 1)/tile_size.y;
	int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1, 0}, dy[] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};

	delete_tile = false;

	switch(state.tiles[index].state) {
		case Tile::RENDER:
		{
			assert(!only_denoise);

			if(!schedule_denoising) {
				state.tiles[index].state = Tile::DONE;
				delete_tile = true;
				return true;
			}
			state.tiles[index].state = Tile::RENDERED;
			/* For each neighbor and the tile itself, check whether all of its neighbors have been rendered. If yes, it can be denoised. */
			for(int n = 0; n < 9; n++) {
				int nx = state.tiles[index].x/tile_size.x + dx[n], ny = state.tiles[index].y/tile_size.y + dy[n];
				if(nx < 0 || ny < 0 || nx >= tile_w || ny >= tile_h)
					continue;
				int nindex = ny*state.tile_stride + nx;
				if(state.tiles[nindex].state != Tile::RENDERED)
					continue;
				bool can_be_denoised = true;
				for(int nn = 0; nn < 8; nn++) {
					int nnx = state.tiles[nindex].x/tile_size.x + dx[nn], nny = state.tiles[nindex].y/tile_size.y + dy[nn];
					if(nnx < 0 || nny < 0 || nnx >= tile_w || nny >= tile_h)
						continue;
					int nnindex = nny*state.tile_stride + nnx;
					if(state.tiles[nnindex].state < Tile::RENDERED) {
						can_be_denoised = false;
						break;
					}
				}
				if(can_be_denoised) {
					state.tiles[nindex].state = Tile::DENOISE;
					state.denoise_tiles[state.tiles[nindex].device].push_back(nindex);
				}
			}
			return false;
		}
		case Tile::DENOISE:
		{
			if(only_denoise) {
				state.tiles[index].state = Tile::DONE;
				delete_tile = false;
				return true;
			}
			state.tiles[index].state = Tile::DENOISED;
			/* For each neighbor and the tile itself, check whether all of its neighbors have been denoised. If yes, it can be freed. */
			for(int n = 0; n < 9; n++) {
				int nx = state.tiles[index].x/tile_size.x + dx[n], ny = state.tiles[index].y/tile_size.y + dy[n];
				if(nx < 0 || ny < 0 || nx >= tile_w || ny >= tile_h)
					continue;
				int nindex = ny*state.tile_stride + nx;
				if(state.tiles[nindex].state != Tile::DENOISED)
					continue;
				bool can_be_freed = true;
				for(int nn = 0; nn < 8; nn++) {
					int nnx = state.tiles[nindex].x/tile_size.x + dx[nn], nny = state.tiles[nindex].y/tile_size.y + dy[nn];
					if(nnx < 0 || nny < 0 || nnx >= tile_w || nny >= tile_h)
						continue;
					int nnindex = nny*state.tile_stride + nnx;
					if(state.tiles[nnindex].state < Tile::DENOISED) {
						can_be_freed = false;
						break;
					}
				}
				if(can_be_freed) {
					state.tiles[nindex].state = Tile::DONE;
					/* It can happen that the tile just finished denoising and already can be freed here.
					 * However, in that case it still has to be written before deleting, so we can't delete it here. */
					if(n == 8) {
						delete_tile = true;
					}
					else {
						delete state.tiles[nindex].buffers;
						state.tiles[nindex].buffers = NULL;
					}
				}
			}
			return true;
		}
		default:
			assert(false);
			return true;
	}
}

bool TileManager::next_tile(Tile* &tile, int device)
{
	int logical_device = preserve_tile_device? device: 0;

	if(logical_device >= state.render_tiles.size())
		return false;

	if(!state.denoise_tiles[logical_device].empty()) {
		int idx = state.denoise_tiles[logical_device].front();
		state.denoise_tiles[logical_device].pop_front();
		tile = &state.tiles[idx];
		if(only_denoise)
			state.num_rendered_tiles++;
		return true;
	}

	if(state.render_tiles[logical_device].empty())
		return false;

	int idx = state.render_tiles[logical_device].front();
	state.render_tiles[logical_device].pop_front();
	tile = &state.tiles[idx];
	state.num_rendered_tiles++;
	return true;
}

bool TileManager::done()
{
	int end_sample = (range_num_samples == -1)
	                     ? num_samples
	                     : range_start_sample + range_num_samples;
	return (state.resolution_divider == 1) &&
	       (state.sample+state.num_samples >= end_sample);
}

bool TileManager::next()
{
	if(done())
		return false;

	if(progressive && state.resolution_divider > 1) {
		state.sample = 0;
		state.resolution_divider /= 2;
		state.num_samples = 1;
		set_tiles();
	}
	else {
		state.sample++;

		if(progressive)
			state.num_samples = 1;
		else if(range_num_samples == -1)
			state.num_samples = num_samples;
		else
			state.num_samples = range_num_samples;

		state.resolution_divider = 1;
		set_tiles();
	}

	return true;
}

int TileManager::get_num_effective_samples()
{
	if(only_denoise) {
		return 1;
	}

	return (range_num_samples == -1) ? num_samples
	                                 : range_num_samples;
}

CCL_NAMESPACE_END

