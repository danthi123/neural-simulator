"""Connectivity generators for the neural simulator.

Standalone functions extracted from SimulationBridge methods.
Each generator builds a CuPy CSR sparse matrix representing
the synaptic weight matrix for a network of *n* neurons.
"""

import time
from collections import defaultdict

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
except ImportError:
    cp = None
    csp = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _calculate_distances_3d_gpu(pos_i_cp, pos_neighbors_cp):
    """Euclidean distances in 3D between a point and an array of other points (CuPy)."""
    if pos_neighbors_cp.size == 0:
        return cp.array([], dtype=cp.float32)
    diff_3d = pos_neighbors_cp - pos_i_cp.reshape(1, 3)
    return cp.sqrt(cp.sum(diff_3d ** 2, axis=1))


# ---------------------------------------------------------------------------
# GPU-vectorized spatial generator (Gumbel-max probabilistic sampling)
# ---------------------------------------------------------------------------

def generate_spatial_connections_gpu(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn=print):
    """Generates connections using fully vectorized GPU operations (fast, scalable to 100K+ neurons).
    Uses chunked processing to avoid OOM errors on large networks.
    """
    log_fn("Generating connections (3D spatial, GPU-vectorized)...")
    start_t = time.time()

    if n == 0:
        return csp.csr_matrix((0, 0), dtype=cp.float32)

    dist_decay = getattr(config, 'connection_distance_decay_factor', 0.01)
    trait_bias = getattr(config, 'trait_connection_bias', 0.5)
    min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight
    k = min(max_connections_per_neuron, n - 1)

    # For very large networks, use chunked processing to avoid memory issues
    # Memory for n x n float32 matrix: n^2 * 4 bytes
    # 20GB limit: sqrt(20e9 / 4) ~ 70k neurons can fit in full matrix
    # Use chunking for n > 15000 to be safe (allows 4GB per chunk with overhead)
    if n > 15000:
        return generate_spatial_connections_chunked(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn)

    # Small enough network - use full vectorization
    # Compute all pairwise distances on GPU (n x n matrix)
    pos = neuron_positions_3d_cp  # Shape: (n, 3)
    pos_i = pos[:, None, :]  # Shape: (n, 1, 3)
    pos_j = pos[None, :, :]  # Shape: (1, n, 3)
    diff = pos_i - pos_j  # Shape: (n, n, 3)
    distances = cp.sqrt(cp.sum(diff ** 2, axis=2))  # Shape: (n, n)

    # Set self-distances to infinity to exclude self-connections
    cp.fill_diagonal(distances, cp.inf)

    # Compute connection probabilities
    prob_dist = cp.exp(-dist_decay * distances)

    # Trait similarity component: multiplicative bias, not additive.
    # Additive bias with top-k selection causes same-type segregation
    # (same-type bonus always wins over distance). Multiplicative bias
    # preserves spatial structure while mildly preferring same-type.
    traits_i = traits_cp[:, None]  # Shape: (n, 1)
    traits_j = traits_cp[None, :]  # Shape: (1, n)
    same_type = (traits_i == traits_j).astype(cp.float32)
    # Scale: same-type gets (1 + trait_bias), cross-type gets 1.0
    prob_trait = 1.0 + same_type * trait_bias

    # Combined probability
    conn_prob = prob_dist * prob_trait  # Shape: (n, n)

    # Gumbel-max trick for GPU-vectorized probabilistic sampling:
    # Adding Gumbel noise to log-probabilities and taking top-k is
    # mathematically equivalent to sampling without replacement from the
    # categorical distribution (Vieira 2014), but runs entirely on GPU.
    # This avoids the same-type segregation of deterministic top-k while
    # being O(n*k) on GPU instead of O(n^2) on CPU.
    log_prob = cp.log(cp.maximum(conn_prob, 1e-30))
    # Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
    gumbel_noise = -cp.log(-cp.log(cp.random.uniform(1e-10, 1.0 - 1e-10,
                                                      size=conn_prob.shape,
                                                      dtype=cp.float32)))
    perturbed = log_prob + gumbel_noise
    # Zero out self-connections
    cp.fill_diagonal(perturbed, -cp.inf)

    # Top-k selection on perturbed scores gives probabilistic sampling
    top_k_indices = cp.argsort(perturbed, axis=1)[:, -k:]  # Shape: (n, k)

    # Generate weights for connections
    weights = cp.random.uniform(min_w, max_w, (n, k)).astype(cp.float32)

    # Convert to COO format
    row_indices = cp.repeat(cp.arange(n), k)  # Shape: (n*k,)
    col_indices = top_k_indices.ravel()  # Shape: (n*k,)
    weights_flat = weights.ravel()  # Shape: (n*k,)

    # Create CSR matrix
    conn_matrix = csp.coo_matrix(
        (weights_flat, (row_indices, col_indices)),
        shape=(n, n),
        dtype=cp.float32
    ).tocsr()

    conn_matrix.sort_indices()
    elapsed = time.time() - start_t
    log_fn(f"Connections (3D Spatial GPU): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
    return conn_matrix


# ---------------------------------------------------------------------------
# Random connections (large networks, no spatial constraint)
# ---------------------------------------------------------------------------

def generate_random_connections_large(n, k, traits_np, trait_bias, min_w, max_w, log_fn=print):
    """Generate random connections for very large networks when spatial constraints don't apply.

    Used when connection_radius exceeds spatial extent, meaning all neurons are
    effectively within connection range of each other. Uses chunked processing
    to avoid memory issues.
    """
    start_t = time.time()
    log_fn(f"Generating random connections for {n} neurons (k={k})...")

    all_rows = []
    all_cols = []
    all_weights = []

    # Process in chunks
    chunk_size = max(1000, n // 100)
    num_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        chunk_n = end_idx - start_idx

        # For each neuron in chunk, randomly select k targets
        for i in range(chunk_n):
            neuron_i = start_idx + i
            trait_i = traits_np[neuron_i]

            # Generate candidate pool (exclude self)
            candidates = np.concatenate([np.arange(0, neuron_i), np.arange(neuron_i + 1, n)])

            # Weight by trait similarity
            candidate_traits = traits_np[candidates]
            weights = np.ones(len(candidates), dtype=np.float32)
            weights[candidate_traits == trait_i] += trait_bias

            # Normalize to probabilities
            probs = weights / weights.sum()

            # Sample k targets
            actual_k = min(k, len(candidates))
            targets = np.random.choice(candidates, size=actual_k, replace=False, p=probs)

            # Generate connection weights
            conn_weights = np.random.uniform(min_w, max_w, actual_k).astype(np.float32)

            all_rows.extend([neuron_i] * actual_k)
            all_cols.extend(targets.tolist())
            all_weights.extend(conn_weights.tolist())

        if num_chunks > 1 and (chunk_idx + 1) % max(1, num_chunks // 10) == 0:
            progress = ((chunk_idx + 1) / num_chunks) * 100
            log_fn(f"Random connection progress: {progress:.1f}%")

    # Create sparse matrix
    row_indices_cp = cp.asarray(np.array(all_rows, dtype=np.int32))
    col_indices_cp = cp.asarray(np.array(all_cols, dtype=np.int32))
    weights_cp = cp.asarray(np.array(all_weights, dtype=np.float32))

    conn_matrix = csp.coo_matrix(
        (weights_cp, (row_indices_cp, col_indices_cp)),
        shape=(n, n),
        dtype=cp.float32
    ).tocsr()

    conn_matrix.sort_indices()
    elapsed = time.time() - start_t
    log_fn(f"Connections (Random Large): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
    return conn_matrix


# ---------------------------------------------------------------------------
# Binned spatial generator (CPU-side, for very large networks)
# ---------------------------------------------------------------------------

def generate_spatial_connections_binned(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn=print):
    """Spatial binning approach for very large networks (>50k neurons).

    Instead of computing distances to all N neurons, we divide the space into bins
    and only compute distances to neurons in nearby bins. This reduces memory from
    O(N) to O(N/num_bins * neighborhood_size), making 100K+ networks feasible.
    """
    log_fn("Generating connections (3D spatial, GPU-binned)...")
    start_t = time.time()

    dist_decay = getattr(config, 'connection_distance_decay_factor', 0.01)
    trait_bias = getattr(config, 'trait_connection_bias', 0.5)
    min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight
    k = min(max_connections_per_neuron, n - 1)

    # Transfer positions to CPU for binning (more efficient for indexing)
    positions_np = cp.asnumpy(neuron_positions_3d_cp)
    traits_np = cp.asnumpy(traits_cp)

    # Get spatial bounds
    pos_min = positions_np.min(axis=0)
    pos_max = positions_np.max(axis=0)
    spatial_extent = pos_max - pos_min + 1e-6  # Avoid zero extent
    max_extent = spatial_extent.max()

    # connection_radius = distance at which probability drops to ~1%
    connection_radius = 4.6 / max(dist_decay, 0.001)

    # If connection_radius exceeds spatial extent, all neurons can connect to all others
    # In this case, use random sampling instead of spatial binning
    if connection_radius >= max_extent:
        log_fn(f"Connection radius ({connection_radius:.1f}) >= spatial extent ({max_extent:.1f}). Using random sampling.")
        return generate_random_connections_large(n, k, traits_np, trait_bias, min_w, max_w, log_fn)

    # Determine bin size based on network size (aim for manageable bins)
    # For 100K neurons, target ~500-1000 neurons per bin = ~100-200 bins total
    target_neurons_per_bin = max(500, n // 200)
    num_bins_total = max(27, n // target_neurons_per_bin)
    num_bins_per_dim = max(3, int(np.cbrt(num_bins_total)))

    bin_size = spatial_extent / num_bins_per_dim

    # Recompute actual num_bins based on bin_size
    num_bins_xyz = np.ceil(spatial_extent / bin_size).astype(int)
    num_bins_xyz = np.maximum(num_bins_xyz, 1)

    # How many bins does connection_radius span?
    avg_bin_size = bin_size.mean()
    neighbor_range = max(1, int(np.ceil(connection_radius / avg_bin_size)))

    # Cap neighbor_range to avoid searching more than half the bins
    max_neighbor_range = num_bins_per_dim // 2
    if neighbor_range > max_neighbor_range:
        log_fn(f"Neighbor range {neighbor_range} too large. Using random sampling.")
        return generate_random_connections_large(n, k, traits_np, trait_bias, min_w, max_w, log_fn)

    log_fn(f"Spatial binning: {num_bins_xyz} bins, bin_size={avg_bin_size:.2f}, neighbor_range={neighbor_range}")

    # Assign each neuron to a bin
    bin_indices = np.floor((positions_np - pos_min) / bin_size).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins_xyz - 1)  # Clamp to valid range

    # Convert 3D bin index to linear index
    bin_linear = (bin_indices[:, 0] * num_bins_xyz[1] * num_bins_xyz[2] +
                  bin_indices[:, 1] * num_bins_xyz[2] +
                  bin_indices[:, 2])

    # Build bin-to-neuron lookup (dict: bin_id -> list of neuron indices)
    bin_to_neurons = defaultdict(list)
    for neuron_idx, bin_id in enumerate(bin_linear):
        bin_to_neurons[bin_id].append(neuron_idx)

    # Pre-compute neighbor offsets based on neighbor_range
    # If neighbor_range=1, we search 3x3x3=27 bins
    # If neighbor_range=2, we search 5x5x5=125 bins, etc.
    neighbor_offsets = []
    for dx in range(-neighbor_range, neighbor_range + 1):
        for dy in range(-neighbor_range, neighbor_range + 1):
            for dz in range(-neighbor_range, neighbor_range + 1):
                neighbor_offsets.append((dx, dy, dz))

    # Process neurons and generate connections - bin-by-bin for vectorization
    all_rows = []
    all_cols = []
    all_weights = []

    # Process bin-by-bin (all neurons in a bin share the same neighbor bins)
    total_bins = len(bin_to_neurons)
    processed_bins = 0

    for bin_id, source_neurons in bin_to_neurons.items():
        if len(source_neurons) == 0:
            continue

        # Get 3D bin coordinates from linear index
        bx = bin_id // (num_bins_xyz[1] * num_bins_xyz[2])
        remainder = bin_id % (num_bins_xyz[1] * num_bins_xyz[2])
        by = remainder // num_bins_xyz[2]
        bz = remainder % num_bins_xyz[2]

        # Gather ALL candidate neurons from neighboring bins (same for all source neurons in this bin)
        candidates = []
        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = bx + dx, by + dy, bz + dz
            if (0 <= nx < num_bins_xyz[0] and
                0 <= ny < num_bins_xyz[1] and
                0 <= nz < num_bins_xyz[2]):
                neighbor_linear = nx * num_bins_xyz[1] * num_bins_xyz[2] + ny * num_bins_xyz[2] + nz
                candidates.extend(bin_to_neurons[neighbor_linear])

        if len(candidates) == 0:
            continue

        # Convert to arrays for vectorized operations
        source_arr = np.array(source_neurons, dtype=np.int32)
        candidate_arr = np.array(candidates, dtype=np.int32)

        # Get positions and traits for sources and candidates
        source_pos = positions_np[source_arr]  # (num_sources, 3)
        candidate_pos = positions_np[candidate_arr]  # (num_candidates, 3)
        source_traits = traits_np[source_arr]  # (num_sources,)
        candidate_traits = traits_np[candidate_arr]  # (num_candidates,)

        # Compute all pairwise distances: (num_sources, num_candidates)
        # Using broadcasting: diff = source_pos[:, None, :] - candidate_pos[None, :, :]
        diff = source_pos[:, None, :] - candidate_pos[None, :, :]  # (S, C, 3)
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (S, C)

        # Set self-distances to infinity
        # Create mask where source[i] == candidate[j]
        source_expanded = source_arr[:, None]  # (S, 1)
        candidate_expanded = candidate_arr[None, :]  # (1, C)
        self_mask = (source_expanded == candidate_expanded)  # (S, C)
        distances[self_mask] = np.inf

        # Compute connection probabilities
        prob_dist = np.exp(-dist_decay * distances)  # (S, C)

        # Trait similarity: multiplicative bias (not additive)
        trait_match = (source_traits[:, None] == candidate_traits[None, :])
        prob_trait = 1.0 + trait_match.astype(np.float32) * trait_bias  # (S, C)

        conn_prob = prob_dist * prob_trait  # (S, C)

        # For each source neuron, select top-k candidates
        num_candidates = len(candidate_arr)
        actual_k = min(k, num_candidates - 1)  # -1 to account for self-exclusion

        if actual_k <= 0:
            continue

        # Use argpartition for each row to get top-k indices
        if actual_k < num_candidates:
            # Partition to get top-k indices (unsorted)
            partition_idx = np.argpartition(conn_prob, -actual_k, axis=1)[:, -actual_k:]  # (S, k)
        else:
            partition_idx = np.tile(np.arange(num_candidates), (len(source_arr), 1))

        # Generate connections
        num_sources = len(source_arr)
        for i in range(num_sources):
            source_neuron = source_arr[i]
            # Filter out any infinite distances (self-connections that might slip through)
            valid_mask = conn_prob[i, partition_idx[i]] > 0
            valid_targets = partition_idx[i][valid_mask]

            if len(valid_targets) == 0:
                continue

            target_neurons = candidate_arr[valid_targets]
            num_connections = len(target_neurons)

            w = np.random.uniform(min_w, max_w, num_connections).astype(np.float32)

            all_rows.extend([source_neuron] * num_connections)
            all_cols.extend(target_neurons.tolist())
            all_weights.extend(w.tolist())

        processed_bins += 1
        if total_bins > 10 and processed_bins % max(1, total_bins // 10) == 0:
            progress = (processed_bins / total_bins) * 100
            log_fn(f"Binned connection progress: {progress:.1f}%")

    # Convert to arrays and create sparse matrix on GPU
    if len(all_rows) == 0:
        log_fn("Warning: No connections generated!")
        return csp.csr_matrix((n, n), dtype=cp.float32)

    row_indices_cp = cp.asarray(np.array(all_rows, dtype=np.int32))
    col_indices_cp = cp.asarray(np.array(all_cols, dtype=np.int32))
    weights_cp = cp.asarray(np.array(all_weights, dtype=np.float32))

    conn_matrix = csp.coo_matrix(
        (weights_cp, (row_indices_cp, col_indices_cp)),
        shape=(n, n),
        dtype=cp.float32
    ).tocsr()

    conn_matrix.sort_indices()
    elapsed = time.time() - start_t
    log_fn(f"Connections (3D Spatial GPU-Binned): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
    return conn_matrix


# ---------------------------------------------------------------------------
# Chunked GPU spatial generator (for 15K-500K neurons)
# ---------------------------------------------------------------------------

def generate_spatial_connections_chunked(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn=print):
    """Chunked version of vectorized connection generation for large networks.
    Processes neurons in GPU-accelerated batches.  Memory-adaptive chunk sizing
    keeps peak VRAM within safe limits for networks up to ~500K neurons on 24GB cards.

    Falls back to the CPU-based binned generator only when a single chunk row
    would exceed available VRAM (extremely large N with high connectivity).
    """
    # Estimate per-chunk-row VRAM: N * 60 bytes (distance matrix + probs + argpartition)
    # Fall back to CPU-binned only if a SINGLE row would exceed 25% of free VRAM
    # (meaning even chunk_size=1 would OOM)
    mem_info = cp.cuda.Device().mem_info
    free_mem = mem_info[0]
    bytes_per_row = n * 60
    if bytes_per_row > free_mem * 0.25:
        log_fn(f"Single chunk row ({bytes_per_row/1e9:.1f}GB) exceeds 25% of free VRAM ({free_mem/1e9:.1f}GB). Falling back to CPU-binned generator.")
        return generate_spatial_connections_binned(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn)

    log_fn("Generating connections (3D spatial, GPU-vectorized-chunked)...")
    start_t = time.time()

    dist_decay = getattr(config, 'connection_distance_decay_factor', 0.01)
    trait_bias = getattr(config, 'trait_connection_bias', 0.5)
    min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight
    k = min(max_connections_per_neuron, n - 1)

    # Determine chunk size based on available memory
    # Peak memory per chunk row (all arrays that coexist during argpartition):
    #   diff:           n * 3 * 4 = 12n bytes  (chunk_n, n, 3) float32
    #   distances:      n * 4     =  4n bytes  (chunk_n, n)    float32
    #   prob_dist:      n * 4     =  4n bytes  (chunk_n, n)    float32
    #   prob_trait:     n * 4     =  4n bytes  (chunk_n, n)    float32
    #   conn_prob:      n * 4     =  4n bytes  (chunk_n, n)    float32
    #   argpartition internals (thrust sort): ~3x (chunk_n, n) int32+float32
    #                   n * 24    = 24n bytes  (hidden CuPy/Thrust temporaries)
    # Total peak: ~52n bytes per chunk row.  Use 60n for safety margin.
    mem_info = cp.cuda.Device().mem_info
    free_mem = mem_info[0]  # Free VRAM in bytes

    # Use only 35% of free memory -- argpartition's Thrust backend allocates
    # large hidden temporaries that are not visible to CuPy's pool accounting
    target_mem_bytes = free_mem * 0.35

    bytes_per_chunk_row = n * 60  # Conservative: accounts for Thrust sort internals
    chunk_size = max(64, int(target_mem_bytes / bytes_per_chunk_row))
    chunk_size = min(chunk_size, n)  # Don't exceed total neurons

    free_mem_gb = free_mem / 1e9
    target_mem_gb = target_mem_bytes / 1e9
    log_fn(f"VRAM: {free_mem_gb:.2f}GB free, using {target_mem_gb:.2f}GB ({target_mem_gb/free_mem_gb*100:.0f}%) for chunking")

    log_fn(f"Using chunked processing: {n} neurons, chunk_size={chunk_size}")

    # Lists to accumulate connection data
    all_rows = []
    all_cols = []
    all_weights = []

    pos = neuron_positions_3d_cp  # Shape: (n, 3)

    # Process neurons in chunks
    num_chunks = (n + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        chunk_n = end_idx - start_idx

        # Get positions and traits for this chunk
        chunk_pos = pos[start_idx:end_idx]  # Shape: (chunk_n, 3)
        chunk_traits = traits_cp[start_idx:end_idx]  # Shape: (chunk_n,)

        # Compute distances from chunk neurons to ALL neurons
        # chunk_pos: (chunk_n, 3) -> (chunk_n, 1, 3)
        # pos: (n, 3) -> (1, n, 3)
        chunk_pos_i = chunk_pos[:, None, :]  # (chunk_n, 1, 3)
        pos_j = pos[None, :, :]  # (1, n, 3)
        diff = chunk_pos_i - pos_j  # (chunk_n, n, 3)
        distances = cp.sqrt(cp.sum(diff ** 2, axis=2))  # (chunk_n, n)

        # Set self-distances to infinity (for neurons in this chunk)
        for i in range(chunk_n):
            global_idx = start_idx + i
            distances[i, global_idx] = cp.inf

        # Compute connection probabilities
        prob_dist = cp.exp(-dist_decay * distances)  # (chunk_n, n)

        # Trait similarity component (multiplicative, not additive)
        chunk_traits_i = chunk_traits[:, None]  # (chunk_n, 1)
        traits_j = traits_cp[None, :]  # (1, n)
        same_type = (chunk_traits_i == traits_j).astype(cp.float32)
        prob_trait = 1.0 + same_type * trait_bias  # (chunk_n, n)

        # Combined probability (multiplicative)
        conn_prob = prob_dist * prob_trait  # (chunk_n, n)

        # Free intermediate arrays BEFORE selection
        del prob_dist, prob_trait, distances, diff
        del chunk_pos_i, pos_j, same_type

        # Gumbel-max trick for probabilistic top-k (avoids same-type segregation)
        log_prob = cp.log(cp.maximum(conn_prob, 1e-30))
        gumbel_noise = -cp.log(-cp.log(cp.random.uniform(
            1e-10, 1.0 - 1e-10, size=conn_prob.shape, dtype=cp.float32)))
        perturbed = log_prob + gumbel_noise
        del log_prob, gumbel_noise, conn_prob
        # Zero out self-connections
        for i in range(chunk_n):
            perturbed[i, start_idx + i] = -cp.inf

        top_k_indices = cp.argsort(perturbed, axis=1)[:, -k:]  # (chunk_n, k)
        del perturbed

        # Generate weights
        weights = cp.random.uniform(min_w, max_w, (chunk_n, k)).astype(cp.float32)

        # Create row indices (offset by start_idx for global indexing)
        chunk_rows = cp.repeat(cp.arange(start_idx, end_idx), k)  # (chunk_n * k,)
        chunk_cols = top_k_indices.ravel()  # (chunk_n * k,)
        chunk_weights = weights.ravel()  # (chunk_n * k,)

        # Accumulate (transfer to CPU immediately to free GPU memory)
        all_rows.append(cp.asnumpy(chunk_rows))
        all_cols.append(cp.asnumpy(chunk_cols))
        all_weights.append(cp.asnumpy(chunk_weights))

        # Explicit cleanup to prevent memory fragmentation
        # (diff, distances, prob_dist, prob_trait, chunk_pos_i, pos_j already freed pre-argpartition)
        del chunk_rows, chunk_cols, chunk_weights, weights
        del top_k_indices, top_k_values, sorted_within_k, partition_idx
        del conn_prob, chunk_pos, chunk_traits
        cp.get_default_memory_pool().free_all_blocks()

        # Progress update (every 10% or every chunk if few chunks)
        if num_chunks > 1 and ((chunk_idx + 1) % max(1, num_chunks // 10) == 0 or chunk_idx == num_chunks - 1):
            progress = ((chunk_idx + 1) / num_chunks) * 100
            elapsed = time.time() - start_t
            eta = elapsed / (chunk_idx + 1) * (num_chunks - chunk_idx - 1)
            log_fn(f"Chunked progress: {progress:.1f}% ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

    # Concatenate all chunks
    all_rows_np = np.concatenate(all_rows)
    all_cols_np = np.concatenate(all_cols)
    all_weights_np = np.concatenate(all_weights)

    # Convert back to GPU and create sparse matrix
    row_indices_cp = cp.asarray(all_rows_np)
    col_indices_cp = cp.asarray(all_cols_np)
    weights_cp = cp.asarray(all_weights_np)

    conn_matrix = csp.coo_matrix(
        (weights_cp, (row_indices_cp, col_indices_cp)),
        shape=(n, n),
        dtype=cp.float32
    ).tocsr()

    conn_matrix.sort_indices()
    elapsed = time.time() - start_t
    log_fn(f"Connections (3D Spatial GPU-Chunked): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
    return conn_matrix


# ---------------------------------------------------------------------------
# Dispatcher: spatial connections (selects vectorized vs legacy)
# ---------------------------------------------------------------------------

def generate_spatial_connections_3d(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn=print):
    """Generates synaptic connections based on spatial proximity and trait similarity in 3D."""
    # Use vectorized GPU version for better performance
    if n > 1000:  # Use vectorized for large networks
        return generate_spatial_connections_gpu(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn)

    # Legacy iterative version for small networks (< 1000 neurons)
    log_fn("Generating connections (3D spatial, legacy)...")
    start_t = time.time()
    if n == 0:
        log_fn("No neurons to connect (n=0).")
        return csp.csr_matrix((0, 0), dtype=cp.float32)

    dist_decay_factor = getattr(config, 'connection_distance_decay_factor', 0.01)
    trait_bias = getattr(config, 'trait_connection_bias', 0.5)
    min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight

    rows, cols, weights_list = [], [], []

    for i in range(n):
        pos_i_cp = neuron_positions_3d_cp[i:i + 1, :]
        trait_i_val = traits_cp[i]

        candidate_indices_np = np.array([j for j in range(n) if j != i], dtype=np.int32)
        if candidate_indices_np.size == 0:
            continue

        candidate_indices_cp = cp.asarray(candidate_indices_np)
        pos_candidates_cp = neuron_positions_3d_cp[candidate_indices_cp]
        traits_candidates_cp = traits_cp[candidate_indices_cp]

        distances_cp = _calculate_distances_3d_gpu(pos_i_cp, pos_candidates_cp)
        prob_distance_component = cp.exp(-dist_decay_factor * distances_cp)
        prob_trait_component = (traits_candidates_cp == trait_i_val).astype(cp.float32) * trait_bias
        connection_probabilities_cp = prob_distance_component + prob_trait_component

        sum_probs = cp.sum(connection_probabilities_cp)
        if sum_probs > 1e-9:
            normalized_probabilities_cp = connection_probabilities_cp / sum_probs
        else:
            if connection_probabilities_cp.size > 0:
                normalized_probabilities_cp = cp.ones_like(connection_probabilities_cp) / connection_probabilities_cp.size
            else:
                continue

        num_potential_targets = candidate_indices_cp.size
        if num_potential_targets > 0:
            num_to_select = min(max_connections_per_neuron, num_potential_targets)

            if num_to_select > 0:
                try:
                    if not np.isclose(cp.asnumpy(cp.sum(normalized_probabilities_cp)), 1.0) and cp.sum(normalized_probabilities_cp) > 1e-9:
                        normalized_probabilities_cp = normalized_probabilities_cp / cp.sum(normalized_probabilities_cp)
                    elif cp.sum(normalized_probabilities_cp) <= 1e-9:
                        selected_local_indices_cp = cp.random.choice(cp.arange(num_potential_targets), size=num_to_select, replace=False)
                    else:
                        selected_local_indices_cp = cp.random.choice(
                            cp.arange(num_potential_targets),
                            size=num_to_select,
                            replace=False,
                            p=normalized_probabilities_cp
                        )
                except (ValueError, NotImplementedError) as e:
                    sorted_local_indices_cp = cp.argsort(connection_probabilities_cp)[::-1]
                    selected_local_indices_cp = sorted_local_indices_cp[:num_to_select]

                final_target_global_indices_cp = candidate_indices_cp[selected_local_indices_cp]
                initial_weights_np = np.random.uniform(min_w, max_w, num_to_select).astype(np.float32)
                final_weights_np = np.clip(initial_weights_np, min_w, max_w)

                rows.extend([i] * num_to_select)
                cols.extend(cp.asnumpy(final_target_global_indices_cp).tolist())
                weights_list.extend(final_weights_np.tolist())

        if n > 0 and i % (max(1, n // 20)) == 0:
            print(f"\rConn gen (3D Spatial): {i / n * 100:.1f}%", end="")

    if n > 0:
        print("\rConn gen (3D Spatial): 100.0% ")

    if not rows:
        log_fn("No connections generated by 3D spatial method.")
        return csp.csr_matrix((n, n), dtype=cp.float32)

    conn_matrix = csp.csr_matrix(
        (cp.asarray(weights_list, dtype=cp.float32),
         (cp.asarray(rows, dtype=cp.int32), cp.asarray(cols, dtype=cp.int32))),
        shape=(n, n), dtype=cp.float32
    )
    conn_matrix.sort_indices()
    log_fn(f"Connections (3D Spatial): {conn_matrix.nnz}. Time: {time.time() - start_t:.2f}s")
    return conn_matrix


# ---------------------------------------------------------------------------
# Watts-Strogatz small-world generator
# ---------------------------------------------------------------------------

def generate_watts_strogatz_3d(n, k_neighbors, p_rewire, config, positions_cp, log_fn=print):
    """Generates connections using a Watts-Strogatz small-world network model in 3D.

    Creates a small-world network with high clustering and short path lengths:
    1. Create ring lattice based on 3D spatial proximity (k nearest neighbors)
    2. Rewire each edge with probability p_rewire to a random target
    3. Maintain directed network structure

    Args:
        n: Number of neurons
        k_neighbors: Number of nearest spatial neighbors to connect (must be even)
        p_rewire: Rewiring probability (0 = regular lattice, 1 = random network)
        config: CoreSimConfig with weight parameters
        positions_cp: CuPy array of neuron 3D positions (n, 3)
        log_fn: Logging callable
    """
    log_fn(f"Generating Watts-Strogatz 3D network (n={n}, k={k_neighbors}, p_rewire={p_rewire})...")
    start_t = time.time()

    if n == 0:
        return csp.csr_matrix((0, 0), dtype=cp.float32)

    if n == 1:
        log_fn("Only 1 neuron, returning empty connectivity.")
        return csp.csr_matrix((1, 1), dtype=cp.float32)

    # Ensure k is valid and even
    k = min(k_neighbors, n - 1)
    if k % 2 == 1:
        k = k + 1  # Make even
        k = min(k, n - 1)
    if k < 2:
        k = 2

    min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight

    # Step 1: Create spatial ordering - sort neurons by 3D position
    # We'll use a space-filling curve approximation (sum of coordinates)
    positions = positions_cp
    spatial_order = cp.sum(positions, axis=1)  # Simple spatial key
    sorted_indices = cp.argsort(spatial_order)

    # Step 2: Build k-nearest neighbor ring lattice
    # Each neuron connects to its k/2 predecessors and k/2 successors in spatial order
    rows = []
    cols = []
    weights = []

    half_k = k // 2

    for i in range(n):
        source_idx = int(sorted_indices[i])

        # Connect to k/2 neighbors on each side in the spatial ring
        for offset in range(1, half_k + 1):
            # Forward connections (clockwise)
            target_spatial_idx = (i + offset) % n
            target_idx = int(sorted_indices[target_spatial_idx])

            # Rewiring decision
            if cp.random.random() < p_rewire:
                # Rewire to random target (avoid self-loops and duplicates)
                target_idx = int(cp.random.randint(0, n))
                while target_idx == source_idx:
                    target_idx = int(cp.random.randint(0, n))

            weight = float(cp.random.uniform(min_w, max_w))
            rows.append(source_idx)
            cols.append(target_idx)
            weights.append(weight)

            # Backward connections (counter-clockwise)
            target_spatial_idx = (i - offset) % n
            target_idx = int(sorted_indices[target_spatial_idx])

            # Rewiring decision
            if cp.random.random() < p_rewire:
                # Rewire to random target
                target_idx = int(cp.random.randint(0, n))
                while target_idx == source_idx:
                    target_idx = int(cp.random.randint(0, n))

            weight = float(cp.random.uniform(min_w, max_w))
            rows.append(source_idx)
            cols.append(target_idx)
            weights.append(weight)

        # Progress indicator for large networks
        if n > 1000 and i % (n // 20) == 0:
            print(f"\rWS generation: {i / n * 100:.1f}%", end="")

    if n > 1000:
        print("\rWS generation: 100.0%")

    # Step 3: Create sparse matrix and remove duplicate edges
    # Convert to COO, then CSR to handle duplicates
    rows_cp = cp.array(rows, dtype=cp.int32)
    cols_cp = cp.array(cols, dtype=cp.int32)
    weights_cp = cp.array(weights, dtype=cp.float32)

    conn_matrix = csp.coo_matrix(
        (weights_cp, (rows_cp, cols_cp)),
        shape=(n, n),
        dtype=cp.float32
    ).tocsr()

    # Remove self-loops if any exist
    conn_matrix.setdiag(cp.zeros(n, dtype=cp.float32))
    conn_matrix.eliminate_zeros()

    conn_matrix.sort_indices()
    elapsed = time.time() - start_t

    # Calculate network statistics
    avg_degree = conn_matrix.nnz / n if n > 0 else 0

    log_fn(
        f"Watts-Strogatz network complete: {conn_matrix.nnz} connections "
        f"(avg degree: {avg_degree:.1f}, expected: {k}). Time: {elapsed:.2f}s"
    )

    return conn_matrix


# ---------------------------------------------------------------------------
# Motif-based connectivity generator
# ---------------------------------------------------------------------------

def generate_motif_connections_3d(n, neuron_positions_3d_cp, traits_cp, config, motif_name, connectivity_motifs, log_fn=print):
    """Generates connections according to a high-level connectivity motif.

    Motifs are defined in CONNECTIVITY_MOTIFS and operate on trait-based
    populations. This generator is optimized for small-to-medium networks
    where explicit population-based sampling is acceptable.

    Args:
        n: Number of neurons
        neuron_positions_3d_cp: CuPy array of 3D positions
        traits_cp: CuPy array of neuron traits
        config: CoreSimConfig
        motif_name: Name of the motif to use
        connectivity_motifs: Dict of motif definitions (CONNECTIVITY_MOTIFS)
        log_fn: Logging callable
    """
    motif_def = connectivity_motifs.get(motif_name)
    if motif_def is None:
        log_fn(f"Unknown connectivity motif '{motif_name}'. Falling back to spatial generator.")
        return generate_spatial_connections_3d(n, config.connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn)

    log_fn(f"Generating connections (Motif: {motif_name})...")
    start_t = time.time()

    if n == 0:
        return csp.csr_matrix((0, 0), dtype=cp.float32)

    # For very large networks, fall back to spatial generator to avoid O(N^2) patterns
    if n > 50000:
        log_fn(
            f"Network size n={n} too large for motif generator; falling back to spatial generator.",
        )
        return generate_spatial_connections_3d(n, config.connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn)

    # Traits on host for flexible population definitions
    if traits_cp is not None and traits_cp.size == n:
        traits_np = cp.asnumpy(traits_cp).astype(np.int32)
    else:
        traits_np = np.zeros(n, dtype=np.int32)

    base_k = getattr(config, "connectivity_k", getattr(config, "connections_per_neuron", 10))
    if base_k < 1:
        base_k = 1
    min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight

    rows: list[int] = []
    cols: list[int] = []
    weights_list: list[float] = []

    rules = motif_def.get("rules", [])
    for rule in rules:
        src_traits = rule.get("source_traits", [])
        tgt_traits = rule.get("target_traits", [])
        if not src_traits or not tgt_traits:
            continue

        k_fraction = float(rule.get("k_fraction", 1.0))
        if k_fraction <= 0.0:
            continue

        weight_scale = float(rule.get("weight_scale", 1.0))

        src_mask = np.isin(traits_np, np.array(src_traits, dtype=np.int32))
        tgt_mask = np.isin(traits_np, np.array(tgt_traits, dtype=np.int32))
        src_indices = np.nonzero(src_mask)[0]
        tgt_indices = np.nonzero(tgt_mask)[0]

        if src_indices.size == 0 or tgt_indices.size == 0:
            continue

        rule_k = int(max(0, round(base_k * k_fraction)))
        if rule_k <= 0:
            continue

        # Local weight range for this rule
        local_min_w = min_w * weight_scale
        local_max_w = max_w * weight_scale
        if local_min_w > local_max_w:
            local_min_w, local_max_w = local_max_w, local_min_w

        for src_idx in src_indices:
            # Avoid self-connections when source and target populations overlap
            if traits_np[src_idx] in tgt_traits and tgt_indices.size > 1:
                available_targets = tgt_indices[tgt_indices != src_idx]
                if available_targets.size == 0:
                    continue
            else:
                available_targets = tgt_indices

            num_targets = min(rule_k, available_targets.size)
            if num_targets <= 0:
                continue

            chosen_targets = np.random.choice(available_targets, size=num_targets, replace=False)
            w = np.random.uniform(local_min_w, local_max_w, size=num_targets).astype(np.float32)

            rows.extend([int(src_idx)] * num_targets)
            cols.extend(chosen_targets.astype(np.int32).tolist())
            weights_list.extend(w.tolist())

    if not rows:
        log_fn(
            f"No connections generated by motif '{motif_name}'. Falling back to spatial generator.",
        )
        return generate_spatial_connections_3d(n, config.connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, log_fn)

    conn_matrix = csp.csr_matrix(
        (
            cp.asarray(weights_list, dtype=cp.float32),
            (cp.asarray(rows, dtype=cp.int32), cp.asarray(cols, dtype=cp.int32)),
        ),
        shape=(n, n),
        dtype=cp.float32,
    )
    conn_matrix.sort_indices()
    elapsed = time.time() - start_t
    log_fn(
        f"Connections (Motif {motif_name}): {conn_matrix.nnz} synapses. Time: {elapsed:.2f}s",
    )
    return conn_matrix
