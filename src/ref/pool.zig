//! Fork-join worker pool for the CPU reference.
//!
//! The reference runs ~170 short parallel regions per forward pass (each a
//! few hundred microseconds), so spawning threads per region costs more than
//! the work. Workers persist: a region is published by bumping `generation`;
//! workers spin on it briefly, then park on a futex. The caller runs the last
//! range itself and waits on `pending`. Futex wakes are issued only when a
//! side is actually parked, so back-to-back regions cost no syscalls.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Atomic = std.atomic.Value;

pub const Pool = struct {
    io: Io,
    gpa: Allocator,
    workers: []std.Thread,

    // Written by the caller once per region, read by workers after they
    // observe the new generation (release/acquire on `generation`).
    job: Job = undefined,
    stop: bool = false,

    // Hot atomics on separate cache lines: workers poll `generation` while
    // the caller polls `pending`.
    generation: Atomic(u32) align(std.atomic.cache_line) = .init(0),
    sleeping_workers: Atomic(u32) align(std.atomic.cache_line) = .init(0),
    pending: Atomic(u32) align(std.atomic.cache_line) = .init(0),
    caller_sleeping: Atomic(u32) align(std.atomic.cache_line) = .init(0),

    const spin_limit = 1 << 16;

    const Job = struct {
        func: *const fn (ctx: *const anyopaque, first: usize, last: usize) void,
        ctx: *const anyopaque,
        n: usize,
        chunk: usize,
    };

    /// Starts `n_threads - 1` workers; the caller is the remaining thread.
    pub fn init(gpa: Allocator, io: Io, n_threads: usize) !*Pool {
        const pool = try gpa.create(Pool);
        errdefer gpa.destroy(pool);
        pool.* = .{ .io = io, .gpa = gpa, .workers = try gpa.alloc(std.Thread, n_threads -| 1) };
        errdefer gpa.free(pool.workers);
        for (pool.workers, 0..) |*w, i| {
            w.* = std.Thread.spawn(.{ .stack_size = 1 << 20 }, workerMain, .{ pool, i }) catch |err| {
                pool.shutdown(pool.workers[0..i]);
                return err;
            };
        }
        return pool;
    }

    pub fn deinit(pool: *Pool) void {
        pool.shutdown(pool.workers);
        pool.gpa.free(pool.workers);
        pool.gpa.destroy(pool);
    }

    pub fn threadCount(pool: *const Pool) usize {
        return pool.workers.len + 1;
    }

    fn shutdown(pool: *Pool, started: []std.Thread) void {
        pool.stop = true;
        _ = pool.generation.fetchAdd(1, .seq_cst);
        pool.io.futexWake(u32, &pool.generation.raw, std.math.maxInt(u32));
        for (started) |t| t.join();
    }

    /// Runs `func(ctx, first, last)` on contiguous ranges covering [0, n).
    /// Range boundaries are multiples of `align_to`; ranges are no smaller
    /// than `min_chunk` unless there is only one. Not reentrant.
    pub fn parallelFor(
        pool: *Pool,
        n: usize,
        align_to: usize,
        min_chunk: usize,
        ctx: anytype,
        comptime func: fn (@TypeOf(ctx), usize, usize) void,
    ) void {
        const Ctx = @TypeOf(ctx);
        const erased = struct {
            fn call(c: *const anyopaque, first: usize, last: usize) void {
                func(@as(Ctx, @ptrCast(@alignCast(c))), first, last);
            }
        };
        const parts = @max(1, @min(pool.threadCount(), n / @max(1, min_chunk)));
        const chunk = std.mem.alignForward(usize, std.math.divCeil(usize, n, parts) catch unreachable, align_to);
        if (parts == 1 or pool.workers.len == 0) {
            func(ctx, 0, n);
            return;
        }

        pool.job = .{ .func = erased.call, .ctx = @ptrCast(ctx), .n = n, .chunk = chunk };
        pool.pending.store(@intCast(pool.workers.len), .seq_cst);
        _ = pool.generation.fetchAdd(1, .seq_cst);
        if (pool.sleeping_workers.load(.seq_cst) != 0) {
            pool.io.futexWake(u32, &pool.generation.raw, std.math.maxInt(u32));
        }

        // The caller takes the range after all workers' ranges.
        runRange(&pool.job, pool.workers.len);

        var spins: usize = 0;
        while (true) {
            const left = pool.pending.load(.acquire);
            if (left == 0) break;
            if (spins < spin_limit) {
                spins += 1;
                std.atomic.spinLoopHint();
                continue;
            }
            pool.caller_sleeping.store(1, .seq_cst);
            if (pool.pending.load(.seq_cst) != 0) {
                pool.io.futexWaitUncancelable(u32, &pool.pending.raw, left);
            }
            pool.caller_sleeping.store(0, .seq_cst);
        }
    }

    fn runRange(job: *const Job, index: usize) void {
        const first = index * job.chunk;
        if (first >= job.n) return;
        job.func(job.ctx, first, @min(job.n, first + job.chunk));
    }

    fn workerMain(pool: *Pool, index: usize) void {
        var seen: u32 = 0;
        while (true) {
            var spins: usize = 0;
            var gen = pool.generation.load(.acquire);
            while (gen == seen) : (gen = pool.generation.load(.acquire)) {
                if (spins < spin_limit) {
                    spins += 1;
                    std.atomic.spinLoopHint();
                    continue;
                }
                _ = pool.sleeping_workers.fetchAdd(1, .seq_cst);
                pool.io.futexWaitUncancelable(u32, &pool.generation.raw, seen);
                _ = pool.sleeping_workers.fetchSub(1, .seq_cst);
            }
            seen = gen;
            if (pool.stop) return;

            runRange(&pool.job, index);

            // seq_cst pairs with the caller's store to `caller_sleeping` then
            // re-load of `pending`: one side always sees the other.
            if (pool.pending.fetchSub(1, .seq_cst) == 1 and pool.caller_sleeping.load(.seq_cst) != 0) {
                pool.io.futexWake(u32, &pool.pending.raw, 1);
            }
        }
    }
};

test "parallelFor covers every index exactly once, across repeated regions" {
    const gpa = std.testing.allocator;
    const pool = try Pool.init(gpa, std.testing.io, 5);
    defer pool.deinit();

    var hits: [1000]Atomic(u32) = @splat(.init(0));
    const Ctx = struct {
        hits: []Atomic(u32),
        fn run(c: *const @This(), first: usize, last: usize) void {
            for (c.hits[first..last]) |*h| _ = h.fetchAdd(1, .monotonic);
        }
    };
    for ([_]usize{ 1, 3, 4, 17, 999, 1000 }) |n| {
        for (0..50) |_| {
            const ctx: Ctx = .{ .hits = hits[0..n] };
            pool.parallelFor(n, 4, 1, &ctx, Ctx.run);
        }
        for (hits[0..n]) |*h| try std.testing.expectEqual(@as(u32, 50), h.swap(0, .monotonic));
    }
}
