#include "computesetintersection.h"
#include <cstdint>

size_t ComputeSetIntersection::galloping_cnt_ = 0;
size_t ComputeSetIntersection::merge_cnt_ = 0;

void ComputeSetIntersection::ComputeCandidates(const VertexID* larray, const ui l_count,
                                               const VertexID* rarray, const ui r_count,
                                               VertexID* cn, ui &cn_count) {
#if HYBRID == 0
    #if SI == 0
    if (l_count / 50 > r_count || r_count / 50 > l_count) {
        galloping_cnt_ += 1;
        return ComputeCNGallopingAVX2(larray, l_count, rarray, r_count, cn, cn_count);
    }
    else {
        merge_cnt_ += 1;
        return ComputeCNMergeBasedAVX2(larray, l_count, rarray, r_count, cn, cn_count);
    }
    #elif SI == 1
    if (l_count / 50 > r_count || r_count / 50 > l_count) {
        galloping_cnt_ += 1;
        return ComputeCNGallopingAVX512(larray, l_count, rarray, r_count, cn, cn_count);
    }
    else {
        merge_cnt_ += 1;
        return ComputeCNMergeBasedAVX512(larray, l_count, rarray, r_count, cn, cn_count);
    }
    #elif SI == 2
    if (l_count / 50 > r_count || r_count / 50 > l_count) {
        galloping_cnt_ += 1;
        return ComputeCNGalloping(larray, l_count, rarray, r_count, cn, cn_count);
    }
    else {
        merge_cnt_ += 1;
        return ComputeCNNaiveStdMerge(larray, l_count, rarray, r_count, cn, cn_count);
    }
    #elif SI == 3
        return MaxStepIntersect(larray, l_count, rarray, r_count, cn, cn_count);
    #endif
#elif HYBRID == 1
    #if SI == 0
        return ComputeCNMergeBasedAVX2(larray, l_count, rarray, r_count, cn, cn_count);
    #elif SI == 1
        return ComputeCNMergeBasedAVX512(larray, l_count, rarray, r_count, cn, cn_count);
    #elif SI == 2
        return ComputeCNNaiveStdMerge(larray, l_count, rarray, r_count, cn, cn_count);
    #elif SI == 3
        return MaxStepIntersect(larray, l_count, rarray, r_count, cn, cn_count);
    #endif
#endif
}

void ComputeSetIntersection::ComputeCandidates(const VertexID* larray, const ui l_count,
                                               const VertexID* rarray, const ui r_count,
                                               ui &cn_count) {
#if HYBRID == 0
    #if SI == 0
        if (l_count / 32 > r_count || r_count / 32 > l_count) {
            return ComputeCNGallopingAVX2(larray, l_count, rarray, r_count, cn_count);
        }
        else {
            return ComputeCNMergeBasedAVX2(larray, l_count, rarray, r_count, cn_count);
        }
    #elif SI == 1
        if (l_count / 32 > r_count || r_count / 32 > l_count) {
            return ComputeCNGallopingAVX512(larray, l_count, rarray, r_count, cn_count);
        }
        else {
            return ComputeCNMergeBasedAVX512(larray, l_count, rarray, r_count, cn_count);
        }
    #elif SI == 2
        if (l_count / 32 > r_count || r_count / 32 > l_count) {
            return ComputeCNGalloping(larray, l_count, rarray, r_count, cn_count);
        }
        else {
            return ComputeCNNaiveStdMerge(larray, l_count, rarray, r_count, cn_count);
        }
    #elif SI == 3
        return MaxStepIntersect(larray, l_count, rarray, r_count, cn_count);
    #endif
#elif HYBRID == 1
    #if SI == 0
        return ComputeCNMergeBasedAVX2(larray, l_count, rarray, r_count, cn_count);
    #elif SI == 1
        return ComputeCNMergeBasedAVX512(larray, l_count, rarray, r_count, cn_count);
    #elif SI == 2
        return ComputeCNNaiveStdMerge(larray, l_count, rarray, r_count, cn_count);
    #elif SI == 3
        return MaxStepIntersect(larray, l_count, rarray, r_count, cn_count);
    #endif
#endif
}

#if SI == 0
void ComputeSetIntersection::ComputeCNGallopingAVX2(const VertexID* larray, const ui l_count,
                                                    const VertexID* rarray, const ui r_count,
                                                    VertexID* cn, ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    while (true) {
        while (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }

        ri = GallopingSearchAVX2(rarray, ri, rc, larray[li]);
        if (ri >= rc) {
            return;
        }

        if (larray[li] == rarray[ri]) {
            cn[cn_count++] = larray[li];
            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

void ComputeSetIntersection::ComputeCNGallopingAVX2(const VertexID* larray, const ui l_count,
                                                    const VertexID* rarray, const ui r_count,
                                                    ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    while (true) {
        while (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }

        ri = GallopingSearchAVX2(rarray, ri, rc, larray[li]);
        if (ri >= rc) {
            return;
        }

        if (larray[li] == rarray[ri]) {
            cn_count += 1;
            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

void ComputeSetIntersection::ComputeCNMergeBasedAVX2(const VertexID* larray, const ui l_count,
                                                     const VertexID* rarray, const ui r_count,
                                                     VertexID* cn, ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    __m256i per_u_order = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
    __m256i per_v_order = _mm256_set_epi32(3, 2, 1, 0, 3, 2, 1, 0);
    VertexID* cur_back_ptr = cn;

    auto size_ratio = (rc) / (lc);
    if (size_ratio > 2) {
        if (li < lc && ri + 7 < rc) {
            __m256i u_elements = _mm256_set1_epi32(larray[li]);
            __m256i v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));

            while (true) {
                __m256i mask = _mm256_cmpeq_epi32(u_elements, v_elements);
                auto real_mask = _mm256_movemask_epi8(mask);
                if (real_mask != 0) {
                    // at most 1 element
                    *cur_back_ptr = larray[li];
                    cur_back_ptr += 1;
                }
                if (larray[li] > rarray[ri + 7]) {
                    ri += 8;
                    if (ri + 7 >= rc) {
                        break;
                    }
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                } else {
                    li++;
                    if (li >= lc) {
                        break;
                    }
                    u_elements = _mm256_set1_epi32(larray[li]);
                }
            }
        }
    } else {
        if (li + 1 < lc && ri + 3 < rc) {
            __m256i u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
            __m256i u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
            __m256i v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
            __m256i v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);

            while (true) {
                __m256i mask = _mm256_cmpeq_epi32(u_elements_per, v_elements_per);
                auto real_mask = _mm256_movemask_epi8(mask);
                if (real_mask << 16 != 0) {
                    *cur_back_ptr = larray[li];
                    cur_back_ptr += 1;
                }
                if (real_mask >> 16 != 0) {
                    *cur_back_ptr = larray[li + 1];
                    cur_back_ptr += 1;
                }


                if (larray[li + 1] == rarray[ri + 3]) {
                    li += 2;
                    ri += 4;
                    if (li + 1 >= lc || ri + 3 >= rc) {
                        break;
                    }
                    u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
                    u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                    v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);
                } else if (larray[li + 1] > rarray[ri + 3]) {
                    ri += 4;
                    if (ri + 3 >= rc) {
                        break;
                    }
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                    v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);
                } else {
                    li += 2;
                    if (li + 1 >= lc) {
                        break;
                    }
                    u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
                    u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
                }
            }
        }
    }

    cn_count = (ui)(cur_back_ptr - cn);
    if (li < lc && ri < rc) {
        while (true) {
            while (larray[li] < rarray[ri]) {
                ++li;
                if (li >= lc) {
                    return;
                }
            }
            while (larray[li] > rarray[ri]) {
                ++ri;
                if (ri >= rc) {
                    return;
                }
            }
            if (larray[li] == rarray[ri]) {
                // write back
                cn[cn_count++] = larray[li];

                ++li;
                ++ri;
                if (li >= lc || ri >= rc) {
                    return;
                }
            }
        }
    }
    return;
}

void ComputeSetIntersection::ComputeCNMergeBasedAVX2(const VertexID* larray, const ui l_count,
                                                     const VertexID* rarray, const ui r_count,
                                                     ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    constexpr int parallelism = 8;

    int cn_countv[parallelism] = {0, 0, 0, 0, 0, 0, 0, 0};
    __m256i sse_cn_countv = _mm256_loadu_si256((__m256i *) (cn_countv));
    __m256i sse_countplus = _mm256_set1_epi32(1);
    __m256i per_u_order = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
    __m256i per_v_order = _mm256_set_epi32(3, 2, 1, 0, 3, 2, 1, 0);

    auto size_ratio = (rc) / (lc);
    if (size_ratio > 2) {
        if (li < lc && ri + 7 < rc) {
            __m256i u_elements = _mm256_set1_epi32(larray[li]);
            __m256i v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));

            while (true) {
                __m256i mask = _mm256_cmpeq_epi32(u_elements, v_elements);
                mask = _mm256_and_si256(sse_countplus, mask);
                sse_cn_countv = _mm256_add_epi32(sse_cn_countv, mask);
                if (larray[li] > rarray[ri + 7]) {
                    ri += 8;
                    if (ri + 7 >= rc) {
                        break;
                    }
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                } else {
                    li++;
                    if (li >= lc) {
                        break;
                    }
                    u_elements = _mm256_set1_epi32(larray[li]);
                }
            }
            _mm256_storeu_si256((__m256i *) cn_countv, sse_cn_countv);
            for (int cn_countvplus : cn_countv) { cn_count += cn_countvplus; }
        }
    } else {
        if (li + 1 < lc && ri + 3 < rc) {
            __m256i u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
            __m256i u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
            __m256i v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
            __m256i v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);

            while (true) {
                __m256i mask = _mm256_cmpeq_epi32(u_elements_per, v_elements_per);
                mask = _mm256_and_si256(sse_countplus, mask);
                sse_cn_countv = _mm256_add_epi32(sse_cn_countv, mask);

                if (larray[li + 1] == rarray[ri + 3]) {
                    li += 2;
                    ri += 4;
                    if (li + 1 >= lc || ri + 3 >= rc) {
                        break;
                    }
                    u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
                    u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                    v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);
                } else if (larray[li + 1] > rarray[ri + 3]) {
                    ri += 4;
                    if (ri + 3 >= rc) {
                        break;
                    }
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                    v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);
                } else {
                    li += 2;
                    if (li + 1 >= lc) {
                        break;
                    }
                    u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
                    u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
                }
            }
        }
        _mm256_storeu_si256((__m256i *) cn_countv, sse_cn_countv);
        for (int cn_countvplus : cn_countv) { cn_count += cn_countvplus; }
    }

    if (li < lc && ri < rc) {
        while (true) {
            while (larray[li] < rarray[ri]) {
                ++li;
                if (li >= lc) {
                    return;
                }
            }
            while (larray[li] > rarray[ri]) {
                ++ri;
                if (ri >= rc) {
                    return;
                }
            }
            if (larray[li] == rarray[ri]) {
                cn_count++;
                ++li;
                ++ri;
                if (li >= lc || ri >= rc) {
                    return;
                }
            }
        }
    }
    return;
}

const ui ComputeSetIntersection::BinarySearchForGallopingSearchAVX2(const VertexID* array, ui offset_beg, ui offset_end, ui val) {
    while (offset_end - offset_beg >= 16) {
        auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_beg) + offset_end) / 2);
        _mm_prefetch((char *) &array[(static_cast<unsigned long>(mid + 1) + offset_end) / 2], _MM_HINT_T0);
        _mm_prefetch((char *) &array[(static_cast<unsigned long>(offset_beg) + mid) / 2], _MM_HINT_T0);
        if (array[mid] == val) {
            return mid;
        } else if (array[mid] < val) {
            offset_beg = mid + 1;
        } else {
            offset_end = mid;
        }
    }

    // linear search fallback, be careful with operator>> and operation+ priority
    __m256i pivot_element = _mm256_set1_epi32(val);
    for (; offset_beg + 7 < offset_end; offset_beg += 8) {
        __m256i elements = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(array + offset_beg));
        __m256i cmp_res = _mm256_cmpgt_epi32(pivot_element, elements);
        int mask = _mm256_movemask_epi8(cmp_res);
        if (mask != 0xffffffff) {
            return offset_beg + (_popcnt32(mask) >> 2);
        }
    }
    if (offset_beg < offset_end) {
        auto left_size = offset_end - offset_beg;
        __m256i elements = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(array + offset_beg));
        __m256i cmp_res = _mm256_cmpgt_epi32(pivot_element, elements);
        int mask = _mm256_movemask_epi8(cmp_res);
        int cmp_mask = 0xffffffff >> ((8 - left_size) << 2);
        mask &= cmp_mask;
        if (mask != cmp_mask) { return offset_beg + (_popcnt32(mask) >> 2); }
    }
    return offset_end;
}

const ui ComputeSetIntersection::GallopingSearchAVX2(const VertexID* array, ui offset_beg, ui offset_end, ui val) {
    if (array[offset_end - 1] < val) {
        return offset_end;
    }

    // linear search
    __m256i pivot_element = _mm256_set1_epi32(val);
    if (offset_end - offset_beg >= 8) {
        __m256i elements = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(array + offset_beg));
        __m256i cmp_res = _mm256_cmpgt_epi32(pivot_element, elements);
        int mask = _mm256_movemask_epi8(cmp_res);
        if (mask != 0xffffffff) { return offset_beg + (_popcnt32(mask) >> 2); }
    } else {
        auto left_size = offset_end - offset_beg;
        __m256i elements = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(array + offset_beg));
        __m256i cmp_res = _mm256_cmpgt_epi32(pivot_element, elements);
        int mask = _mm256_movemask_epi8(cmp_res);
        int cmp_mask = 0xffffffff >> ((8 - left_size) << 2);
        mask &= cmp_mask;
        if (mask != cmp_mask) { return offset_beg + (_popcnt32(mask) >> 2); }
    }

    // galloping, should add pre-fetch later
    auto jump_idx = 8u;
    while (true) {
        auto peek_idx = offset_beg + jump_idx;
        if (peek_idx >= offset_end) {
            return BinarySearchForGallopingSearchAVX2(array, (jump_idx >> 1) + offset_beg + 1, offset_end, val);
        }
        if (array[peek_idx] < val) {
            jump_idx <<= 1;
        } else {
            return array[peek_idx] == val ? peek_idx :
                   BinarySearchForGallopingSearchAVX2(array, (jump_idx >> 1) + offset_beg + 1, peek_idx + 1, val);
        }
    }
}

#elif SI == 1
void ComputeSetIntersection::ComputeCNGallopingAVX512(const VertexID* larray, const ui l_count,
                                                          const VertexID* rarray, const ui r_count,
                                                          VertexID* cn, ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    while (true) {
        while (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }

        ri = Utility::GallopingSearchAVX512(rarray, ri, rc, larray[li]);
        if (ri >= rc) {
            return;
        }

        if (larray[li] == rarray[ri]) {
            cn[cn_count++] = larray[li];
            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

void ComputeSetIntersection::ComputeCNGallopingAVX512(const VertexID* larray, const ui l_count,
                                                          const VertexID* rarray, const ui r_count,
                                                          ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    while (true) {
        while (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }

        ri = Utility::GallopingSearchAVX512(rarray, ri, rc, larray[li]);
        if (ri >= rc) {
            return;
        }

        if (larray[li] == rarray[ri]) {
            cn_count += 1;
            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

void ComputeSetIntersection::ComputeCNMergeBasedAVX512(const VertexID* larray, const ui l_count,
                                                       const VertexID* rarray, const ui r_count,
                                                       VertexID* cn, ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    __m512i st = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

    VertexID* cur_back_ptr = cn;

    auto size1 = (rc) / (lc);
    if (size1 > 2) {
        if (li < lc && ri + 15 < rc) {
            __m512i u_elements = _mm512_set1_epi32(larray[li]);
            __m512i v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));

            while (true) {
                __mmask16 mask = _mm512_cmpeq_epi32_mask(u_elements, v_elements);
                if (mask != 0x0000) {
                    // write back
                    _mm512_mask_compressstoreu_epi32(cur_back_ptr, mask, u_elements);
                    cur_back_ptr += _popcnt32(mask);
                }

                if (larray[li] > rarray[ri + 15]) {
                    ri += 16;
                    if (ri + 15 >= rc) {
                        break;
                    }
                    v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));
                } else {
                    li += 1;
                    if (li >= lc) {
                        break;
                    }
                    u_elements = _mm512_set1_epi32(larray[li]);
                }
            }
        }
    } else {
        if (li + 3 < lc && ri + 3 < rc) {
            __m512i u_elements = _mm512_loadu_si512((__m512i *) (larray + li));
            __m512i u_elements_per = _mm512_permutevar_epi32(st, u_elements);
            __m512i v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));
            __m512i v_elements_per = _mm512_permute4f128_epi32(v_elements, 0b00000000);

            while (true) {
                __mmask16 mask = _mm512_cmpeq_epi32_mask(u_elements_per, v_elements_per);
                if (mask != 0x0000) {
                    // write back
                    _mm512_mask_compressstoreu_epi32(cur_back_ptr, mask, u_elements_per);
                    cur_back_ptr += _popcnt32(mask);
                }

                if (larray[li + 3] > rarray[ri + 3]) {
                    ri += 4;
                    if (ri + 3 >= rc) {
                        break;
                    }
                    v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));
                    v_elements_per = _mm512_permute4f128_epi32(v_elements, 0b00000000);
                } else if (larray[li + 3] < rarray[ri + 3]) {
                    li += 4;
                    if (li + 3 >= lc) {
                        break;
                    }
                    u_elements = _mm512_loadu_si512((__m512i *) (larray + li));
                    u_elements_per = _mm512_permutevar_epi32(st, u_elements);
                } else {
                    li += 4;
                    ri += 4;
                    if (li + 3 >= lc || ri + 3 >= rc) {
                        break;
                    }
                    u_elements = _mm512_loadu_si512((__m512i *) (larray + li));
                    u_elements_per = _mm512_permutevar_epi32(st, u_elements);
                    v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));
                    v_elements_per = _mm512_permute4f128_epi32(v_elements, 0b00000000);
                }
            }
        }
    }

    cn_count = (ui)(cur_back_ptr - cn);

    if (li < lc && ri < rc) {
        while (true) {
            while (larray[li] < rarray[ri]) {
                li += 1;
                if (li >= lc) {
                    return;
                }
            }
            while (larray[li] > rarray[ri]) {
                ri += 1;
                if (ri >= rc) {
                    return;
                }
            }
            if (larray[li] == rarray[ri]) {
                // write back
                cn[cn_count++] = larray[li];

                li += 1;
                ri += 1;
                if (li >= lc || ri >= rc) {
                    return;
                }
            }
        }
    }
    return;
}

void ComputeSetIntersection::ComputeCNMergeBasedAVX512(const VertexID* larray, const ui l_count,
                                                       const VertexID* rarray, const ui r_count,
                                                       ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    constexpr int parallelism = 16;
    __m512i st = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    __m512i ssecountplus = _mm512_set1_epi32(1);
    int cn_countv[parallelism] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    __m512i ssecn_countv = _mm512_set1_epi32(0);
    auto size1 = (rc) / (lc);

    if (size1 > 2) {
        if (li < lc && ri + 15 < rc) {
            __m512i u_elements = _mm512_set1_epi32(larray[li]);
            __m512i v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));

            while (true) {
                __mmask16 mask = _mm512_cmpeq_epi32_mask(u_elements, v_elements);
                ssecn_countv = _mm512_mask_add_epi32(ssecn_countv, mask, ssecn_countv, ssecountplus);

                if (larray[li] > rarray[ri + 15]) {
                    ri += 16;
                    if (ri + 15 >= rc) {
                        break;
                    }
                    v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));
                } else {
                    li += 1;
                    if (li >= lc) {
                        break;
                    }
                    u_elements = _mm512_set1_epi32(larray[li]);
                }
            }
            _mm512_storeu_si512((__m512i *) cn_countv, ssecn_countv);
            for (int cn_countvplus : cn_countv) { cn_count += cn_countvplus; }
        }
    } else {
        if (li + 3 < lc && ri + 3 < rc) {
            __m512i u_elements = _mm512_loadu_si512((__m512i *) (larray + li));
            __m512i u_elements_per = _mm512_permutevar_epi32(st, u_elements);
            __m512i v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));
            __m512i v_elements_per = _mm512_permute4f128_epi32(v_elements, 0b00000000);

            while (true) {
                __mmask16 mask = _mm512_cmpeq_epi32_mask(u_elements_per, v_elements_per);
                ssecn_countv = _mm512_mask_add_epi32(ssecn_countv, mask, ssecn_countv, ssecountplus);

                if (larray[li + 3] > rarray[ri + 3]) {
                    ri += 4;
                    if (ri + 3 >= rc) {
                        break;
                    }
                    v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));
                    v_elements_per = _mm512_permute4f128_epi32(v_elements, 0b00000000);
                } else if (larray[li + 3] < rarray[ri + 3]) {
                    li += 4;
                    if (li + 3 >= lc) {
                        break;
                    }
                    u_elements = _mm512_loadu_si512((__m512i *) (larray + li));
                    u_elements_per = _mm512_permutevar_epi32(st, u_elements);
                } else {
                    li += 4;
                    ri += 4;
                    if (li + 3 >= lc || ri + 3 >= rc) {
                        break;
                    }
                    u_elements = _mm512_loadu_si512((__m512i *) (larray + li));
                    u_elements_per = _mm512_permutevar_epi32(st, u_elements);
                    v_elements = _mm512_loadu_si512((__m512i *) (rarray + ri));
                    v_elements_per = _mm512_permute4f128_epi32(v_elements, 0b00000000);
                }
            }
            _mm512_storeu_si512((__m512i *) cn_countv, ssecn_countv);
            for (int cn_countvplus : cn_countv) { cn_count += cn_countvplus; }
        }
    }

    if (li < lc && ri < rc) {
        while (true) {
            while (larray[li] < rarray[ri]) {
                li += 1;
                if (li >= lc) {
                    return;
                }
            }
            while (larray[li] > rarray[ri]) {
                ri += 1;
                if (ri >= rc) {
                    return;
                }
            }
            if (larray[li] == rarray[ri]) {
                cn_count += 1;
                li += 1;
                ri += 1;
                if (li >= lc || ri >= rc) {
                    return;
                }
            }
        }
    }
}

#elif SI == 2
void ComputeSetIntersection::ComputeCNNaiveStdMerge(const VertexID* larray, const ui l_count,
                                                    const VertexID* rarray, const ui r_count,
                                                    VertexID* cn, ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    while (true) {
        if (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }
        else if (larray[li] > rarray[ri]) {
            ri += 1;
            if (ri >= rc) {
                return;
            }
        }
        else {
            cn[cn_count++] = larray[li];

            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

void ComputeSetIntersection::ComputeCNNaiveStdMerge(const VertexID* larray, const ui l_count,
                                                    const VertexID* rarray, const ui r_count,
                                                    ui &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    ui lc = l_count;
    ui rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    while (true) {
        if (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }
        else if (larray[li] > rarray[ri]) {
            ri += 1;
            if (ri >= rc) {
                return;
            }
        }
        else {
            cn_count += 1;
            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

void ComputeSetIntersection::ComputeCNGalloping(const VertexID* larray, const ui l_count,
                                                const VertexID* rarray, const ui r_count,
                                                VertexID* cn, ui &cn_count) {
    ui lc = l_count;
    ui rc = r_count;
    cn_count = 0;
    if (lc == 0 || rc == 0)
        return;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    while (true) {
        while (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }

        ri = GallopingSearch(rarray, ri, rc, larray[li]);
        if (ri >= rc) {
            return;
        }

        if (larray[li] == rarray[ri]) {
            cn[cn_count++] = larray[li];

            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

void ComputeSetIntersection::ComputeCNGalloping(const VertexID* larray, const ui l_count,
                                                const VertexID* rarray, const ui r_count,
                                                ui &cn_count) {
    ui lc = l_count;
    ui rc = r_count;
    cn_count = 0;
    if (lc == 0 || rc == 0)
        return;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        ui tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    ui li = 0;
    ui ri = 0;

    while (true) {
        while (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }

        ri = GallopingSearch(rarray, ri, rc, larray[li]);
        if (ri >= rc) {
            return;
        }

        if (larray[li] == rarray[ri]) {
            cn_count += 1;

            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

const ui ComputeSetIntersection::GallopingSearch(const VertexID *src, const ui begin, const ui end,
                                            const ui target) {
    if (src[end - 1] < target) {
        return end;
    }
    // galloping
    if (src[begin] >= target) {
        return begin;
    }
    if (src[begin + 1] >= target) {
        return begin + 1;
    }
    if (src[begin + 2] >= target) {
        return begin + 2;
    }

    ui jump_idx = 4;
    ui offset_beg = begin;
    while (true) {
        ui peek_idx = offset_beg + jump_idx;
        if (peek_idx >= end) {
            return BinarySearch(src, (jump_idx >> 1) + offset_beg + 1, end, target);
        }
        if (src[peek_idx] < target) {
            jump_idx <<= 1;
        } else {
            return src[peek_idx] == target ? peek_idx :
                   BinarySearch(src, (jump_idx >> 1) + offset_beg + 1, peek_idx + 1, target);
        }
    }
}

const ui ComputeSetIntersection::BinarySearch(const VertexID *src, const ui begin, const ui end, const ui target) {
    int offset_begin = begin;
    int offset_end = end;
    while (offset_end - offset_begin >= 16) {
        auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_begin) + offset_end) / 2);
        _mm_prefetch((char *) &src[(mid + 1 + offset_end) / 2], _MM_HINT_T0);
        _mm_prefetch((char *) &src[(mid - 1 + offset_begin) / 2], _MM_HINT_T0);
        if (src[mid] == target) {
            return mid;
        } else if (src[mid] < target) {
            offset_begin = mid + 1;
        } else {
            offset_end = mid;
        }
    }

    // linear search fallback
    for (auto offset = offset_begin; offset < offset_end; ++offset) {
        if (src[offset] >= target) {
            return (ui)offset;
        }
    }

    return (ui)offset_end;
}
#elif SI == 3

size_t binary_search(VertexID const *array, size_t start, size_t length, VertexID val) {
    size_t low = start;
    size_t high = start + 1;
    while (high < length && array[high] < val) {
        low = high;
        high = (2 * high < length) ? 2 * high : length;
    }
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (array[mid] < val) { low = mid + 1; }
        else { high = mid; }
    }
    return low;
}

void intersect_linear(VertexID const *l, size_t l_count, VertexID const *r, size_t r_count, VertexID *results, ui &result_count) {
    result_count = 0;
    size_t i = 0, j = 0;
    while (i != l_count && j != r_count) {
        uint32_t li = l[i];
        uint32_t rj = r[j];
        results[result_count] = li;
        result_count += li == rj;
        i += li  < rj;
        j += li >= rj;
    }
}

void intersect_linear(VertexID const *l, size_t l_count, VertexID const *r, size_t r_count, ui &result_count) {
    result_count = 0;
    size_t i = 0, j = 0;
    while (i != l_count && j != r_count) {
        uint32_t li = l[i];
        uint32_t rj = r[j];
        result_count += li == rj;
        i += li  < rj;
        j += li >= rj;
    }
}

void intersect_serial(VertexID const *l, size_t l_count, VertexID const *r, size_t r_count, VertexID *results, ui &result_count) {
    /* Swap arrays if necessary, as we want "longer" to be larger than "shorter" */
    if (r_count < l_count) {
        uint32_t const *temp = l;
        l = r;
        r = temp;
        size_t temp_length = l_count;
        l_count = r_count;
        r_count = temp_length;
    }

    /* Use the accurate implementation if galloping is not beneficial */
    if (r_count < 64 * l_count) {
        intersect_linear(l, l_count, r, r_count, results, result_count);
        return;
    }

    /* Perform galloping, shrinking the target range */
    result_count = 0;
    size_t j = 0;
    for (size_t i = 0; i < l_count; ++i) {
        uint32_t li = l[i];
        j = binary_search(r, j, r_count, li);
        if (j < r_count && r[j] == li) { 
            results[result_count] = li; 
            result_count++; 
        }
    }
}

void intersect_serial(VertexID const *l, size_t l_count, VertexID const *r, size_t r_count, ui &result_count) {
    /* Swap arrays if necessary, as we want "longer" to be larger than "shorter" */
    if (r_count < l_count) {
        uint32_t const *temp = l;
        l = r;
        r = temp;
        size_t temp_length = l_count;
        l_count = r_count;
        r_count = temp_length;
    }

    /* Use the accurate implementation if galloping is not beneficial */
    if (r_count < 64 * l_count) {
        intersect_linear(l, l_count, r, r_count, result_count);
        return;
    }

    /* Perform galloping, shrinking the target range */
    result_count = 0;
    size_t j = 0;
    for (size_t i = 0; i < l_count; ++i) {
        uint32_t li = l[i];
        j = binary_search(r, j, r_count, li);
        if (j < r_count && r[j] == li) {
            result_count++; 
        }
    }
}

__mmask16 rol(__mmask16 x, int n) { return (x << n) | (x >> (16 - n)); }
__mmask16 ror(__mmask16 x, int n) { return (x >> n) | (x << (16 - n)); }

__mmask16 simd_intersect(__m512i l, __m512i r) {
    __m512i l1 = _mm512_alignr_epi32(l, l, 4);
    __m512i r1 = _mm512_shuffle_epi32(r, _MM_PERM_ADCB);
    __mmask16 nm00 = _mm512_cmpneq_epi32_mask(l, r);

    __m512i l2 = _mm512_alignr_epi32(l, l, 8);
    __m512i l3 = _mm512_alignr_epi32(l, l, 12);
    __mmask16 nm01 = _mm512_cmpneq_epi32_mask(l1, r);
    __mmask16 nm02 = _mm512_cmpneq_epi32_mask(l2, r);

    __mmask16 nm03 = _mm512_cmpneq_epi32_mask(l3, r);
    __mmask16 nm10 = _mm512_mask_cmpneq_epi32_mask(nm00, l, r1);
    __mmask16 nm11 = _mm512_mask_cmpneq_epi32_mask(nm01, l1, r1);

    __m512i r2 = _mm512_shuffle_epi32(r, _MM_PERM_BADC);
    __mmask16 nm12 = _mm512_mask_cmpneq_epi32_mask(nm02, l2, r1);
    __mmask16 nm13 = _mm512_mask_cmpneq_epi32_mask(nm03, l3, r1);
    __mmask16 nm20 = _mm512_mask_cmpneq_epi32_mask(nm10, l, r2);

    __m512i r3 = _mm512_shuffle_epi32(r, _MM_PERM_CBAD);
    __mmask16 nm21 = _mm512_mask_cmpneq_epi32_mask(nm11, l1, r2);
    __mmask16 nm22 = _mm512_mask_cmpneq_epi32_mask(nm12, l2, r2);
    __mmask16 nm23 = _mm512_mask_cmpneq_epi32_mask(nm13, l3, r2);

    __mmask16 nm0 = _mm512_mask_cmpneq_epi32_mask(nm20, l, r3);
    __mmask16 nm1 = _mm512_mask_cmpneq_epi32_mask(nm21, l1, r3);
    __mmask16 nm2 = _mm512_mask_cmpneq_epi32_mask(nm22, l2, r3);
    __mmask16 nm3 = _mm512_mask_cmpneq_epi32_mask(nm23, l3, r3);

    return ~(__mmask16) (nm0 & rol(nm1, 4) & rol(nm2, 8) & ror(nm3, 4));
}

void ComputeSetIntersection::MaxStepIntersect(const VertexID* l, ui l_count, const VertexID* r, ui r_count, VertexID* results, ui &result_count) {
    // Optimization for very small sets
    if (l_count < 16 && r_count < 16) {
        intersect_serial(l, l_count, r, r_count, results, result_count);
        return;
    }

    result_count = 0;
    uint32_t const *const l_end = l + l_count;
    uint32_t const *const r_end = r + r_count;
    ui c = 0;
    union vec_t {
        __m512i zmm;
        uint32_t u32[16];
        uint8_t u8[64];
    } l_vec, r_vec;

    while (l + 16 < l_end && r + 16 < r_end) {
        l_vec.zmm = _mm512_loadu_si512((__m512i const *) l);
        r_vec.zmm = _mm512_loadu_si512((__m512i const *) r);

        // Intersecting registers with involves a lot of shuffling and comparisons, 
        // so we want to avoid it if the slices don't overlap at all..
        uint32_t l_min;
        uint32_t l_max = l_vec.u32[15];
        uint32_t r_min = r_vec.u32[0];
        uint32_t r_max = r_vec.u32[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (l_max < r_min && l + 32 < l_end) {
            l += 16;
            l_vec.zmm = _mm512_loadu_si512((__m512i const *) l);
            l_max = l_vec.u32[15];
        }
        l_min = l_vec.u32[0];
        while (r_max < l_min && r + 32 < r_end) {
            r += 16;
            r_vec.zmm = _mm512_loadu_si512((__m512i const *) r);
            r_max = r_vec.u32[15];
        }
        r_min = r_vec.u32[0];

        __m512i l_max_vec = _mm512_set1_epi32(l_max);
        __m512i r_max_vec = _mm512_set1_epi32(r_max);
        __mmask16 l_step_mask = _mm512_cmple_epu32_mask(l_vec.zmm, r_max_vec);
        __mmask16 r_step_mask = _mm512_cmple_epu32_mask(r_vec.zmm, l_max_vec);
        l += 32 - _lzcnt_u32((uint32_t) l_step_mask);
        r += 32 - _lzcnt_u32((uint32_t) r_step_mask);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask16 matches = simd_intersect(l_vec.zmm, r_vec.zmm);

        // Write matches
        _mm512_mask_compressstoreu_epi32(results + result_count, matches, l_vec.zmm);
        result_count += _mm_popcnt_u32(matches);
    }

    // Handle tail
    intersect_serial(l, l_end - l, r, r_end - r, results + result_count, c);
    result_count += c;
}

void ComputeSetIntersection::MaxStepIntersect(const VertexID* l, ui l_count, const VertexID* r, ui r_count, ui &result_count) {
    // Optimization for very small sets
    if (l_count < 16 && r_count < 16) {
        intersect_serial(l, l_count, r, r_count, result_count);
        return;
    }

    result_count = 0;
    uint32_t const *const l_end = l + l_count;
    uint32_t const *const r_end = r + r_count;
    ui c = 0;
    union vec_t {
        __m512i zmm;
        uint32_t u32[16];
        uint8_t u8[64];
    } l_vec, r_vec;

    while (l + 16 < l_end && r + 16 < r_end) {
        l_vec.zmm = _mm512_loadu_si512((__m512i const *) l);
        r_vec.zmm = _mm512_loadu_si512((__m512i const *) r);

        // Intersecting registers with involves a lot of shuffling and comparisons, 
        // so we want to avoid it if the slices don't overlap at all..
        uint32_t l_min;
        uint32_t l_max = l_vec.u32[15];
        uint32_t r_min = r_vec.u32[0];
        uint32_t r_max = r_vec.u32[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (l_max < r_min && l + 32 < l_end) {
            l += 16;
            l_vec.zmm = _mm512_loadu_si512((__m512i const *) l);
            l_max = l_vec.u32[15];
        }
        l_min = l_vec.u32[0];
        while (r_max < l_min && r + 32 < r_end) {
            r += 16;
            r_vec.zmm = _mm512_loadu_si512((__m512i const *) r);
            r_max = r_vec.u32[15];
        }
        r_min = r_vec.u32[0];

        __m512i l_max_vec = _mm512_set1_epi32(l_max);
        __m512i r_max_vec = _mm512_set1_epi32(r_max);
        __mmask16 l_step_mask = _mm512_cmple_epu32_mask(l_vec.zmm, r_max_vec);
        __mmask16 r_step_mask = _mm512_cmple_epu32_mask(r_vec.zmm, l_max_vec);
        l += 32 - _lzcnt_u32((uint32_t) l_step_mask);
        r += 32 - _lzcnt_u32((uint32_t) r_step_mask);

        __mmask16 matches = simd_intersect(l_vec.zmm, r_vec.zmm);

        result_count += _mm_popcnt_u32(matches);
    }

    // Handle tail
    intersect_serial(l, l_end - l, r, r_end - r, c);
    result_count += c;
}

#endif
