# run_pipeline_debug.py
import numpy as np
import traceback, time, os

# 修改下面这行为你的模块名（如果不是 fgp_functions_new.py）
import fgp_functions_new as fe

def save_dbg(arr, name_prefix):
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        fname = f"{name_prefix}_pid{pid}_{ts}.npy"
        np.save(fname, np.asarray(arr))
        print(f"[DBG] saved {name_prefix} -> {fname}, shape={np.asarray(arr).shape}")
        return fname
    except Exception as e:
        print("[DBG] failed to save", e)
        return None

def info(obj, tag):
    try:
        a = np.asarray(obj)
        print(f"{tag}: type={type(obj)}, shape={getattr(a,'shape',None)}, size={getattr(a,'size',None)}")
    except Exception as e:
        print(f"{tag}: cannot convert to ndarray: {e}")

def check_and_save(arr, tag):
    try:
        a = np.asarray(arr)
    except Exception:
        print(f"[CHECK] {tag}: cannot convert to ndarray")
        traceback.print_stack(limit=6)
        return
    info(arr, tag)
    if a.size == 0 or any((d == 0 for d in a.shape if isinstance(d, int))):
        print(f"[CHECK] {tag} is empty or has zero-dim -> saving and dumping stack")
        save_dbg(a, tag)
        traceback.print_stack(limit=8)

def simulate_pipeline(img1=None, img2=None):
    # prepare synthetic images if not provided
    if img1 is None:
        img1 = (np.arange(32*32, dtype=np.float32) % 256).reshape(32,32)
    if img2 is None:
        img2 = (np.random.randint(0,256,size=(32,32))).astype(np.float32)

    print("=== Starting pipeline simulation ===")
    info(img1, "Image1 (input)")
    info(img2, "Image2 (input)")

    try:
        # Branch A: deep chain on Image2
        print("\n--- Branch A: RegionR(Image2, 19, 6, 26, 28) ---")
        regionA = fe.regionR(img2, 19, 6, 26, 28)
        check_and_save(regionA, "regionA_after_regionR")

        # first Median: use median_raw to get Img2 format (features, s list)
        print("\n--- Applying Median (median_raw) with s=1 ---")
        med0 = fe.median_raw(regionA, 1)   # returns [features, signal_f]
        print("med0 (median_raw) -> returned container types:")
        print("  med0[0] (features) len:", len(med0[0]), " med0[1] (signal) len:", len(med0[1]))
        if len(med0[0])>0:
            info(med0[0][0], "med0.features[0]")

        # now successive Median2 calls (medianf) with s values: 0,0,0,1 per your nesting
        curr = med0
        for idx, s_val in enumerate([0, 0, 0, 1], start=1):
            print(f"\n--- Branch A: Median2 step {idx} with s={s_val} ---")
            curr = fe.medianf(curr, s_val)  # curr stays in Img2 format: [features, signals]
            # inspect curr
            try:
                feats = curr[0]
                sigs = curr[1]
            except Exception:
                print("  medianf returned unexpected structure:", type(curr))
                info(curr, f"medianf_result_step{idx}")
                break
            print(f"  medianf result: features len={len(feats)}; signals len={len(sigs)}")
            if len(feats)>0:
                info(feats[0], f"medianf_result_step{idx}.features[0]")
                check_and_save(feats[0], f"medianf_step{idx}_features0")

        # Now call uniform_lbp on the final Img2 result (curr)
        print("\n--- Branch A: uniform_lbp on final Median2 result ---")
        check_and_save(curr, "branchA_before_uniform_lbp_container")
        # uniform_lbp expects img2 format (x, s2) which curr already is
        uA = fe.uniform_lbp(curr)
        print("uniform_lbp returned features count:", len(uA))
        if len(uA)>0:
            info(uA[0], "uniform_lbp branchA feature[0]")

    except Exception as e:
        print("Exception in Branch A:", e)
        traceback.print_exc()

    try:
        # Branch B: RegionR(Image1, 13, 11, 17, 22) -> Median -> uLBP
        print("\n--- Branch B: RegionR(Image1, 13, 11, 17, 22) ---")
        regionB = fe.regionR(img1, 13, 11, 17, 22)
        check_and_save(regionB, "regionB_after_regionR")

        print("\n--- Branch B: Median (median_raw) with s=0 ---")
        medB = fe.median_raw(regionB, 0)  # returns [features, signal_f]
        print("medB: features len", len(medB[0]), "signals len", len(medB[1]))
        if len(medB[0])>0:
            info(medB[0][0], "medB.features[0]")
            check_and_save(medB[0][0], "medB.features.0")

        print("\n--- Branch B: uniform_lbp on medB ---")
        uB = fe.uniform_lbp(medB)
        print("uniform_lbp branchB returned count:", len(uB))
        if len(uB)>0:
            info(uB[0], "uniform_lbp branchB feature[0]")

    except Exception as e:
        print("Exception in Branch B:", e)
        traceback.print_exc()

    # Final combine (Features2C)
    try:
        print("\n--- Combine: feature_con( branchA_result, branchB_result ) ---")
        # uA and uB are lists of features; feature_con expects vectors; if uA/uB contain lists, flatten:
        # Use first element of each uniform_lbp as representative (as original pipeline likely does)
        vA = uA[0] if (isinstance(uA, (list,tuple)) and len(uA)>0) else np.array([])
        vB = uB[0] if (isinstance(uB, (list,tuple)) and len(uB)>0) else np.array([])
        info(vA, "vA (to feature_con)")
        info(vB, "vB (to feature_con)")
        combined = fe.feature_con(vA, vB)
        print("feature_con result shape:", getattr(np.asarray(combined),'shape',None))
    except Exception as e:
        print("Exception combining features:", e)
        traceback.print_exc()

    print("\n=== Pipeline simulation finished ===")

if __name__ == "__main__":
    simulate_pipeline()
