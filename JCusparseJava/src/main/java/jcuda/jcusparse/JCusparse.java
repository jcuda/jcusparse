/*
 * JCusparse - Java bindings for CUSPARSE, the NVIDIA CUDA sparse
 * matrix library, to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

package jcuda.jcusparse;

import jcuda.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

/**
 * Java bindings for CUSPARSE, the NVIDIA CUDA sparse matrix
 * BLAS library.
 */
public class JCusparse
{
    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not
     * cusparseStatus.CUSPARSE_STATUS_SUCCESS
     */
    private static boolean exceptionsEnabled = false;

    /* Private constructor to prevent instantiation */
    private JCusparse()
    {
    }

    // Initialize the native library.
    static
    {
        initialize();
    }

    /**
     * Initializes the native library. Note that this method
     * does not have to be called explicitly, since it will
     * be called automatically when this class is loaded.
     */
    public static void initialize()
    {
        if (!initialized)
        {
            String libraryBaseName = "JCusparse-" + JCuda.getJCudaVersion();
            String libraryName = 
                LibUtils.createPlatformLibraryName(libraryBaseName);
            LibUtils.loadLibrary(libraryName);
            initialized = true;
        }
    }


    /**
     * Set the specified log level for the JCusparse library.<br />
     * <br />
     * Currently supported log levels:
     * <br />
     * LOG_QUIET: Never print anything <br />
     * LOG_ERROR: Print error messages <br />
     * LOG_TRACE: Print a trace of all native function calls <br />
     *
     * @param logLevel The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel)
    {
        setLogLevelNative(logLevel.ordinal());
    }

    private static native void setLogLevelNative(int logLevel);


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only set the {@link cusparseStatus} from the native methods.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to set a result code
     * that is not cusparseStatus.CUSPARSE_STATUS_SUCCESS
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is not cusparseStatus.CUSPARSE_STATUS_SUCCESS
     * and exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not cusparseStatus.CUSPARSE_STATUS_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result !=
            cusparseStatus.CUSPARSE_STATUS_SUCCESS)
        {
            throw new CudaException(cusparseStatus.stringFor(result));
        }
        return result;
    }

    /**
     * If the given result is <strong>equal</strong> to
     * cusparseStatus.JCUSPARSE_STATUS_INTERNAL_ERROR
     * and exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.<br />
     * <br />
     * This method is used for the functions that do not return
     * an error code, but a constant value, like a cusparseFillMode.
     * The respective functions may still return internal errors
     * from the JNI part.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is cusparseStatus.JCUSPARSE_STATUS_INTERNAL_ERROR
     */
    private static int checkForError(int result)
    {
        if (exceptionsEnabled && result ==
            cusparseStatus.JCUSPARSE_STATUS_INTERNAL_ERROR)
        {
            throw new CudaException(cusparseStatus.stringFor(result));
        }
        return result;
    }





    /** CUSPARSE initialization and managment routines */
    public static int cusparseCreate(
        cusparseHandle handle)
    {
        return checkResult(cusparseCreateNative(handle));
    }
    private static native int cusparseCreateNative(
        cusparseHandle handle);


    public static int cusparseDestroy(
        cusparseHandle handle)
    {
        return checkResult(cusparseDestroyNative(handle));
    }
    private static native int cusparseDestroyNative(
        cusparseHandle handle);


    public static int cusparseGetVersion(
        cusparseHandle handle, 
        int[] version)
    {
        return checkResult(cusparseGetVersionNative(handle, version));
    }
    private static native int cusparseGetVersionNative(
        cusparseHandle handle, 
        int[] version);


    public static int cusparseGetProperty(
        int type, 
        int[] value)
    {
        return checkResult(cusparseGetPropertyNative(type, value));
    }
    private static native int cusparseGetPropertyNative(
        int type, 
        int[] value);


    public static String cusparseGetErrorName(
        int status)
    {
        return cusparseGetErrorNameNative(status);
    }
    private static native String cusparseGetErrorNameNative(
        int status);


    public static String cusparseGetErrorString(
        int status)
    {
        return cusparseGetErrorStringNative(status);
    }
    private static native String cusparseGetErrorStringNative(
        int status);


    public static int cusparseSetStream(
        cusparseHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cusparseSetStreamNative(handle, streamId));
    }
    private static native int cusparseSetStreamNative(
        cusparseHandle handle, 
        cudaStream_t streamId);


    public static int cusparseGetStream(
        cusparseHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cusparseGetStreamNative(handle, streamId));
    }
    private static native int cusparseGetStreamNative(
        cusparseHandle handle, 
        cudaStream_t streamId);


    public static int cusparseGetPointerMode(
        cusparseHandle handle, 
        int[] mode)
    {
        return checkResult(cusparseGetPointerModeNative(handle, mode));
    }
    private static native int cusparseGetPointerModeNative(
        cusparseHandle handle, 
        int[] mode);


    public static int cusparseSetPointerMode(
        cusparseHandle handle, 
        int mode)
    {
        return checkResult(cusparseSetPointerModeNative(handle, mode));
    }
    private static native int cusparseSetPointerModeNative(
        cusparseHandle handle, 
        int mode);


    //##############################################################################
    //# HELPER ROUTINES
    //##############################################################################
    public static int cusparseCreateMatDescr(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseCreateMatDescrNative(descrA));
    }
    private static native int cusparseCreateMatDescrNative(
        cusparseMatDescr descrA);


    public static int cusparseDestroyMatDescr(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseDestroyMatDescrNative(descrA));
    }
    private static native int cusparseDestroyMatDescrNative(
        cusparseMatDescr descrA);


    public static int cusparseSetMatType(
        cusparseMatDescr descrA, 
        int type)
    {
        return checkResult(cusparseSetMatTypeNative(descrA, type));
    }
    private static native int cusparseSetMatTypeNative(
        cusparseMatDescr descrA, 
        int type);


    public static int cusparseGetMatType(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseGetMatTypeNative(descrA));
    }
    private static native int cusparseGetMatTypeNative(
        cusparseMatDescr descrA);


    public static int cusparseSetMatFillMode(
        cusparseMatDescr descrA, 
        int fillMode)
    {
        return checkResult(cusparseSetMatFillModeNative(descrA, fillMode));
    }
    private static native int cusparseSetMatFillModeNative(
        cusparseMatDescr descrA, 
        int fillMode);


    public static int cusparseGetMatFillMode(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseGetMatFillModeNative(descrA));
    }
    private static native int cusparseGetMatFillModeNative(
        cusparseMatDescr descrA);


    public static int cusparseSetMatDiagType(
        cusparseMatDescr descrA, 
        int diagType)
    {
        return checkResult(cusparseSetMatDiagTypeNative(descrA, diagType));
    }
    private static native int cusparseSetMatDiagTypeNative(
        cusparseMatDescr descrA, 
        int diagType);


    public static int cusparseGetMatDiagType(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseGetMatDiagTypeNative(descrA));
    }
    private static native int cusparseGetMatDiagTypeNative(
        cusparseMatDescr descrA);


    public static int cusparseSetMatIndexBase(
        cusparseMatDescr descrA, 
        int base)
    {
        return checkResult(cusparseSetMatIndexBaseNative(descrA, base));
    }
    private static native int cusparseSetMatIndexBaseNative(
        cusparseMatDescr descrA, 
        int base);


    public static int cusparseGetMatIndexBase(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseGetMatIndexBaseNative(descrA));
    }
    private static native int cusparseGetMatIndexBaseNative(
        cusparseMatDescr descrA);


    public static int cusparseCreateCsrsv2Info(
        csrsv2Info info)
    {
        return checkResult(cusparseCreateCsrsv2InfoNative(info));
    }
    private static native int cusparseCreateCsrsv2InfoNative(
        csrsv2Info info);


    public static int cusparseDestroyCsrsv2Info(
        csrsv2Info info)
    {
        return checkResult(cusparseDestroyCsrsv2InfoNative(info));
    }
    private static native int cusparseDestroyCsrsv2InfoNative(
        csrsv2Info info);


    public static int cusparseCreateCsric02Info(
        csric02Info info)
    {
        return checkResult(cusparseCreateCsric02InfoNative(info));
    }
    private static native int cusparseCreateCsric02InfoNative(
        csric02Info info);


    public static int cusparseDestroyCsric02Info(
        csric02Info info)
    {
        return checkResult(cusparseDestroyCsric02InfoNative(info));
    }
    private static native int cusparseDestroyCsric02InfoNative(
        csric02Info info);


    public static int cusparseCreateBsric02Info(
        bsric02Info info)
    {
        return checkResult(cusparseCreateBsric02InfoNative(info));
    }
    private static native int cusparseCreateBsric02InfoNative(
        bsric02Info info);


    public static int cusparseDestroyBsric02Info(
        bsric02Info info)
    {
        return checkResult(cusparseDestroyBsric02InfoNative(info));
    }
    private static native int cusparseDestroyBsric02InfoNative(
        bsric02Info info);


    public static int cusparseCreateCsrilu02Info(
        csrilu02Info info)
    {
        return checkResult(cusparseCreateCsrilu02InfoNative(info));
    }
    private static native int cusparseCreateCsrilu02InfoNative(
        csrilu02Info info);


    public static int cusparseDestroyCsrilu02Info(
        csrilu02Info info)
    {
        return checkResult(cusparseDestroyCsrilu02InfoNative(info));
    }
    private static native int cusparseDestroyCsrilu02InfoNative(
        csrilu02Info info);


    public static int cusparseCreateBsrilu02Info(
        bsrilu02Info info)
    {
        return checkResult(cusparseCreateBsrilu02InfoNative(info));
    }
    private static native int cusparseCreateBsrilu02InfoNative(
        bsrilu02Info info);


    public static int cusparseDestroyBsrilu02Info(
        bsrilu02Info info)
    {
        return checkResult(cusparseDestroyBsrilu02InfoNative(info));
    }
    private static native int cusparseDestroyBsrilu02InfoNative(
        bsrilu02Info info);


    public static int cusparseCreateBsrsv2Info(
        bsrsv2Info info)
    {
        return checkResult(cusparseCreateBsrsv2InfoNative(info));
    }
    private static native int cusparseCreateBsrsv2InfoNative(
        bsrsv2Info info);


    public static int cusparseDestroyBsrsv2Info(
        bsrsv2Info info)
    {
        return checkResult(cusparseDestroyBsrsv2InfoNative(info));
    }
    private static native int cusparseDestroyBsrsv2InfoNative(
        bsrsv2Info info);


    public static int cusparseCreateBsrsm2Info(
        bsrsm2Info info)
    {
        return checkResult(cusparseCreateBsrsm2InfoNative(info));
    }
    private static native int cusparseCreateBsrsm2InfoNative(
        bsrsm2Info info);


    public static int cusparseDestroyBsrsm2Info(
        bsrsm2Info info)
    {
        return checkResult(cusparseDestroyBsrsm2InfoNative(info));
    }
    private static native int cusparseDestroyBsrsm2InfoNative(
        bsrsm2Info info);


    public static int cusparseCreateCsru2csrInfo(
        csru2csrInfo info)
    {
        return checkResult(cusparseCreateCsru2csrInfoNative(info));
    }
    private static native int cusparseCreateCsru2csrInfoNative(
        csru2csrInfo info);


    public static int cusparseDestroyCsru2csrInfo(
        csru2csrInfo info)
    {
        return checkResult(cusparseDestroyCsru2csrInfoNative(info));
    }
    private static native int cusparseDestroyCsru2csrInfoNative(
        csru2csrInfo info);


    public static int cusparseCreateColorInfo(
        cusparseColorInfo info)
    {
        return checkResult(cusparseCreateColorInfoNative(info));
    }
    private static native int cusparseCreateColorInfoNative(
        cusparseColorInfo info);


    public static int cusparseDestroyColorInfo(
        cusparseColorInfo info)
    {
        return checkResult(cusparseDestroyColorInfoNative(info));
    }
    private static native int cusparseDestroyColorInfoNative(
        cusparseColorInfo info);


    public static int cusparseCreatePruneInfo(
        pruneInfo info)
    {
        return checkResult(cusparseCreatePruneInfoNative(info));
    }
    private static native int cusparseCreatePruneInfoNative(
        pruneInfo info);


    public static int cusparseDestroyPruneInfo(
        pruneInfo info)
    {
        return checkResult(cusparseDestroyPruneInfoNative(info));
    }
    private static native int cusparseDestroyPruneInfoNative(
        pruneInfo info);


    //##############################################################################
    //# SPARSE LEVEL 1 ROUTINES
    //##############################################################################
    @Deprecated
    public static int cusparseSaxpyi(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseSaxpyiNative(handle, nnz, alpha, xVal, xInd, y, idxBase));
    }
    private static native int cusparseSaxpyiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    @Deprecated
    public static int cusparseDaxpyi(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseDaxpyiNative(handle, nnz, alpha, xVal, xInd, y, idxBase));
    }
    private static native int cusparseDaxpyiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    @Deprecated
    public static int cusparseCaxpyi(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseCaxpyiNative(handle, nnz, alpha, xVal, xInd, y, idxBase));
    }
    private static native int cusparseCaxpyiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    @Deprecated
    public static int cusparseZaxpyi(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseZaxpyiNative(handle, nnz, alpha, xVal, xInd, y, idxBase));
    }
    private static native int cusparseZaxpyiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    @Deprecated
    public static int cusparseSgthr(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseSgthrNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseSgthrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    @Deprecated
    public static int cusparseDgthr(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseDgthrNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseDgthrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    @Deprecated
    public static int cusparseCgthr(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseCgthrNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseCgthrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    @Deprecated
    public static int cusparseZgthr(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseZgthrNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseZgthrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    @Deprecated
    public static int cusparseSgthrz(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseSgthrzNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseSgthrzNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    @Deprecated
    public static int cusparseDgthrz(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseDgthrzNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseDgthrzNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    @Deprecated
    public static int cusparseCgthrz(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseCgthrzNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseCgthrzNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    @Deprecated
    public static int cusparseZgthrz(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseZgthrzNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseZgthrzNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    @Deprecated
    public static int cusparseSsctr(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseSsctrNative(handle, nnz, xVal, xInd, y, idxBase));
    }
    private static native int cusparseSsctrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    @Deprecated
    public static int cusparseDsctr(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseDsctrNative(handle, nnz, xVal, xInd, y, idxBase));
    }
    private static native int cusparseDsctrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    @Deprecated
    public static int cusparseCsctr(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseCsctrNative(handle, nnz, xVal, xInd, y, idxBase));
    }
    private static native int cusparseCsctrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    @Deprecated
    public static int cusparseZsctr(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseZsctrNative(handle, nnz, xVal, xInd, y, idxBase));
    }
    private static native int cusparseZsctrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    @Deprecated
    public static int cusparseSroti(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer c, 
        Pointer s, 
        int idxBase)
    {
        return checkResult(cusparseSrotiNative(handle, nnz, xVal, xInd, y, c, s, idxBase));
    }
    private static native int cusparseSrotiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer c, 
        Pointer s, 
        int idxBase);


    @Deprecated
    public static int cusparseDroti(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer c, 
        Pointer s, 
        int idxBase)
    {
        return checkResult(cusparseDrotiNative(handle, nnz, xVal, xInd, y, c, s, idxBase));
    }
    private static native int cusparseDrotiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer c, 
        Pointer s, 
        int idxBase);


    //##############################################################################
    //# SPARSE LEVEL 2 ROUTINES
    //##############################################################################
    public static int cusparseSgemvi(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer beta, 
        Pointer y, 
        int idxBase, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgemviNative(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer));
    }
    private static native int cusparseSgemviNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer beta, 
        Pointer y, 
        int idxBase, 
        Pointer pBuffer);


    public static int cusparseSgemvi_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        int[] pBufferSize)
    {
        return checkResult(cusparseSgemvi_bufferSizeNative(handle, transA, m, n, nnz, pBufferSize));
    }
    private static native int cusparseSgemvi_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        int[] pBufferSize);


    public static int cusparseDgemvi(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer beta, 
        Pointer y, 
        int idxBase, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgemviNative(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer));
    }
    private static native int cusparseDgemviNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer beta, 
        Pointer y, 
        int idxBase, 
        Pointer pBuffer);


    public static int cusparseDgemvi_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        int[] pBufferSize)
    {
        return checkResult(cusparseDgemvi_bufferSizeNative(handle, transA, m, n, nnz, pBufferSize));
    }
    private static native int cusparseDgemvi_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        int[] pBufferSize);


    public static int cusparseCgemvi(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer beta, 
        Pointer y, 
        int idxBase, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgemviNative(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer));
    }
    private static native int cusparseCgemviNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer beta, 
        Pointer y, 
        int idxBase, 
        Pointer pBuffer);


    public static int cusparseCgemvi_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        int[] pBufferSize)
    {
        return checkResult(cusparseCgemvi_bufferSizeNative(handle, transA, m, n, nnz, pBufferSize));
    }
    private static native int cusparseCgemvi_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        int[] pBufferSize);


    public static int cusparseZgemvi(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer beta, 
        Pointer y, 
        int idxBase, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgemviNative(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer));
    }
    private static native int cusparseZgemviNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer beta, 
        Pointer y, 
        int idxBase, 
        Pointer pBuffer);


    public static int cusparseZgemvi_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        int[] pBufferSize)
    {
        return checkResult(cusparseZgemvi_bufferSizeNative(handle, transA, m, n, nnz, pBufferSize));
    }
    private static native int cusparseZgemvi_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        int[] pBufferSize);


    public static int cusparseCsrmvEx_bufferSize(
        cusparseHandle handle, 
        int alg, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        int alphatype, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        int csrValAtype, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        int xtype, 
        Pointer beta, 
        int betatype, 
        Pointer y, 
        int ytype, 
        int executiontype, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseCsrmvEx_bufferSizeNative(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, bufferSizeInBytes));
    }
    private static native int cusparseCsrmvEx_bufferSizeNative(
        cusparseHandle handle, 
        int alg, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        int alphatype, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        int csrValAtype, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        int xtype, 
        Pointer beta, 
        int betatype, 
        Pointer y, 
        int ytype, 
        int executiontype, 
        long[] bufferSizeInBytes);


    public static int cusparseCsrmvEx(
        cusparseHandle handle, 
        int alg, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        int alphatype, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        int csrValAtype, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        int xtype, 
        Pointer beta, 
        int betatype, 
        Pointer y, 
        int ytype, 
        int executiontype, 
        Pointer buffer)
    {
        return checkResult(cusparseCsrmvExNative(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, buffer));
    }
    private static native int cusparseCsrmvExNative(
        cusparseHandle handle, 
        int alg, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        int alphatype, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        int csrValAtype, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        int xtype, 
        Pointer beta, 
        int betatype, 
        Pointer y, 
        int ytype, 
        int executiontype, 
        Pointer buffer);


    public static int cusparseSbsrmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseSbsrmvNative(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseSbsrmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseDbsrmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseDbsrmvNative(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseDbsrmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseCbsrmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseCbsrmvNative(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseCbsrmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseZbsrmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseZbsrmvNative(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseZbsrmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseSbsrxmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedMaskPtrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedEndPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseSbsrxmvNative(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseSbsrxmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedMaskPtrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedEndPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseDbsrxmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedMaskPtrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedEndPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseDbsrxmvNative(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseDbsrxmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedMaskPtrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedEndPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseCbsrxmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedMaskPtrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedEndPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseCbsrxmvNative(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseCbsrxmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedMaskPtrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedEndPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseZbsrxmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedMaskPtrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedEndPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseZbsrxmvNative(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseZbsrxmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedMaskPtrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedEndPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseXcsrsv2_zeroPivot(
        cusparseHandle handle, 
        csrsv2Info info, 
        Pointer position)
    {
        return checkResult(cusparseXcsrsv2_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXcsrsv2_zeroPivotNative(
        cusparseHandle handle, 
        csrsv2Info info, 
        Pointer position);


    public static int cusparseScsrsv2_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseScsrsv2_bufferSizeNative(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseScsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseDcsrsv2_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsrsv2_bufferSizeNative(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseDcsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseCcsrsv2_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsrsv2_bufferSizeNative(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseCcsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseZcsrsv2_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsrsv2_bufferSizeNative(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseZcsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseScsrsv2_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrsv2_analysisNative(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsrsv2_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrsv2_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrsv2_analysisNative(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsrsv2_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrsv2_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrsv2_analysisNative(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsrsv2_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrsv2_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrsv2_analysisNative(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsrsv2_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseScsrsv2_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrsv2_solveNative(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer));
    }
    private static native int cusparseScsrsv2_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrsv2_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrsv2_solveNative(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer));
    }
    private static native int cusparseDcsrsv2_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrsv2_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrsv2_solveNative(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer));
    }
    private static native int cusparseCcsrsv2_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrsv2_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrsv2_solveNative(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer));
    }
    private static native int cusparseZcsrsv2_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseXbsrsv2_zeroPivot(
        cusparseHandle handle, 
        bsrsv2Info info, 
        Pointer position)
    {
        return checkResult(cusparseXbsrsv2_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXbsrsv2_zeroPivotNative(
        cusparseHandle handle, 
        bsrsv2Info info, 
        Pointer position);


    public static int cusparseSbsrsv2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSbsrsv2_bufferSizeNative(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseSbsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseDbsrsv2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDbsrsv2_bufferSizeNative(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseDbsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseCbsrsv2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCbsrsv2_bufferSizeNative(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseCbsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseZbsrsv2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZbsrsv2_bufferSizeNative(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseZbsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseSbsrsv2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrsv2_analysisNative(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseSbsrsv2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrsv2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrsv2_analysisNative(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseDbsrsv2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrsv2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrsv2_analysisNative(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseCbsrsv2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrsv2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrsv2_analysisNative(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseZbsrsv2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSbsrsv2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrsv2_solveNative(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer));
    }
    private static native int cusparseSbsrsv2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrsv2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrsv2_solveNative(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer));
    }
    private static native int cusparseDbsrsv2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrsv2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrsv2_solveNative(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer));
    }
    private static native int cusparseCbsrsv2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrsv2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrsv2_solveNative(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer));
    }
    private static native int cusparseZbsrsv2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer f, 
        Pointer x, 
        int policy, 
        Pointer pBuffer);


    //##############################################################################
    //# SPARSE LEVEL 3 ROUTINES
    //##############################################################################
    public static int cusparseSbsrmm(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseSbsrmmNative(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc));
    }
    private static native int cusparseSbsrmmNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseDbsrmm(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseDbsrmmNative(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc));
    }
    private static native int cusparseDbsrmmNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseCbsrmm(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseCbsrmmNative(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc));
    }
    private static native int cusparseCbsrmmNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseZbsrmm(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseZbsrmmNative(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc));
    }
    private static native int cusparseZbsrmmNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    @Deprecated
    public static int cusparseSgemmi(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer cscValB, 
        Pointer cscColPtrB, 
        Pointer cscRowIndB, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseSgemmiNative(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc));
    }
    private static native int cusparseSgemmiNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer cscValB, 
        Pointer cscColPtrB, 
        Pointer cscRowIndB, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    @Deprecated
    public static int cusparseDgemmi(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer cscValB, 
        Pointer cscColPtrB, 
        Pointer cscRowIndB, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseDgemmiNative(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc));
    }
    private static native int cusparseDgemmiNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer cscValB, 
        Pointer cscColPtrB, 
        Pointer cscRowIndB, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    @Deprecated
    public static int cusparseCgemmi(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer cscValB, 
        Pointer cscColPtrB, 
        Pointer cscRowIndB, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseCgemmiNative(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc));
    }
    private static native int cusparseCgemmiNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer cscValB, 
        Pointer cscColPtrB, 
        Pointer cscRowIndB, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    @Deprecated
    public static int cusparseZgemmi(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer cscValB, 
        Pointer cscColPtrB, 
        Pointer cscRowIndB, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseZgemmiNative(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc));
    }
    private static native int cusparseZgemmiNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        Pointer A, 
        int lda, 
        Pointer cscValB, 
        Pointer cscColPtrB, 
        Pointer cscRowIndB, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseCreateCsrsm2Info(
        csrsm2Info info)
    {
        return checkResult(cusparseCreateCsrsm2InfoNative(info));
    }
    private static native int cusparseCreateCsrsm2InfoNative(
        csrsm2Info info);


    public static int cusparseDestroyCsrsm2Info(
        csrsm2Info info)
    {
        return checkResult(cusparseDestroyCsrsm2InfoNative(info));
    }
    private static native int cusparseDestroyCsrsm2InfoNative(
        csrsm2Info info);


    public static int cusparseXcsrsm2_zeroPivot(
        cusparseHandle handle, 
        csrsm2Info info, 
        Pointer position)
    {
        return checkResult(cusparseXcsrsm2_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXcsrsm2_zeroPivotNative(
        cusparseHandle handle, 
        csrsm2Info info, 
        Pointer position);


    public static int cusparseScsrsm2_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        long[] pBufferSize)
    {
        return checkResult(cusparseScsrsm2_bufferSizeExtNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize));
    }
    private static native int cusparseScsrsm2_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        long[] pBufferSize);


    public static int cusparseDcsrsm2_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        long[] pBufferSize)
    {
        return checkResult(cusparseDcsrsm2_bufferSizeExtNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize));
    }
    private static native int cusparseDcsrsm2_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        long[] pBufferSize);


    public static int cusparseCcsrsm2_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        long[] pBufferSize)
    {
        return checkResult(cusparseCcsrsm2_bufferSizeExtNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize));
    }
    private static native int cusparseCcsrsm2_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        long[] pBufferSize);


    public static int cusparseZcsrsm2_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        long[] pBufferSize)
    {
        return checkResult(cusparseZcsrsm2_bufferSizeExtNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize));
    }
    private static native int cusparseZcsrsm2_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        long[] pBufferSize);


    public static int cusparseScsrsm2_analysis(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrsm2_analysisNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer));
    }
    private static native int cusparseScsrsm2_analysisNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrsm2_analysis(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrsm2_analysisNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer));
    }
    private static native int cusparseDcsrsm2_analysisNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrsm2_analysis(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrsm2_analysisNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer));
    }
    private static native int cusparseCcsrsm2_analysisNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrsm2_analysis(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrsm2_analysisNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer));
    }
    private static native int cusparseZcsrsm2_analysisNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseScsrsm2_solve(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrsm2_solveNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer));
    }
    private static native int cusparseScsrsm2_solveNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrsm2_solve(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrsm2_solveNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer));
    }
    private static native int cusparseDcsrsm2_solveNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrsm2_solve(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrsm2_solveNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer));
    }
    private static native int cusparseCcsrsm2_solveNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrsm2_solve(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrsm2_solveNative(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer));
    }
    private static native int cusparseZcsrsm2_solveNative(
        cusparseHandle handle, 
        int algo, 
        int transA, 
        int transB, 
        int m, 
        int nrhs, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer B, 
        int ldb, 
        csrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseXbsrsm2_zeroPivot(
        cusparseHandle handle, 
        bsrsm2Info info, 
        Pointer position)
    {
        return checkResult(cusparseXbsrsm2_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXbsrsm2_zeroPivotNative(
        cusparseHandle handle, 
        bsrsm2Info info, 
        Pointer position);


    public static int cusparseSbsrsm2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSbsrsm2_bufferSizeNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes));
    }
    private static native int cusparseSbsrsm2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseDbsrsm2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDbsrsm2_bufferSizeNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes));
    }
    private static native int cusparseDbsrsm2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseCbsrsm2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCbsrsm2_bufferSizeNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes));
    }
    private static native int cusparseCbsrsm2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseZbsrsm2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZbsrsm2_bufferSizeNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes));
    }
    private static native int cusparseZbsrsm2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseSbsrsm2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrsm2_analysisNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer));
    }
    private static native int cusparseSbsrsm2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrsm2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrsm2_analysisNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer));
    }
    private static native int cusparseDbsrsm2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrsm2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrsm2_analysisNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer));
    }
    private static native int cusparseCbsrsm2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrsm2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrsm2_analysisNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer));
    }
    private static native int cusparseZbsrsm2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSbsrsm2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer B, 
        int ldb, 
        Pointer X, 
        int ldx, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrsm2_solveNative(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer));
    }
    private static native int cusparseSbsrsm2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer B, 
        int ldb, 
        Pointer X, 
        int ldx, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrsm2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer B, 
        int ldb, 
        Pointer X, 
        int ldx, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrsm2_solveNative(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer));
    }
    private static native int cusparseDbsrsm2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer B, 
        int ldb, 
        Pointer X, 
        int ldx, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrsm2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer B, 
        int ldb, 
        Pointer X, 
        int ldx, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrsm2_solveNative(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer));
    }
    private static native int cusparseCbsrsm2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer B, 
        int ldb, 
        Pointer X, 
        int ldx, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrsm2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer B, 
        int ldb, 
        Pointer X, 
        int ldx, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrsm2_solveNative(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer));
    }
    private static native int cusparseZbsrsm2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer B, 
        int ldb, 
        Pointer X, 
        int ldx, 
        int policy, 
        Pointer pBuffer);


    //##############################################################################
    //# PRECONDITIONERS
    //##############################################################################
    public static int cusparseScsrilu02_numericBoost(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseScsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseScsrilu02_numericBoostNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseDcsrilu02_numericBoost(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseDcsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseDcsrilu02_numericBoostNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseCcsrilu02_numericBoost(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseCcsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseCcsrilu02_numericBoostNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseZcsrilu02_numericBoost(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseZcsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseZcsrilu02_numericBoostNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseXcsrilu02_zeroPivot(
        cusparseHandle handle, 
        csrilu02Info info, 
        Pointer position)
    {
        return checkResult(cusparseXcsrilu02_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXcsrilu02_zeroPivotNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        Pointer position);


    public static int cusparseScsrilu02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseScsrilu02_bufferSizeNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseScsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseDcsrilu02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsrilu02_bufferSizeNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseDcsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseCcsrilu02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsrilu02_bufferSizeNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseCcsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseZcsrilu02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsrilu02_bufferSizeNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseZcsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseScsrilu02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrilu02_analysisNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsrilu02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrilu02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrilu02_analysisNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsrilu02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrilu02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrilu02_analysisNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsrilu02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrilu02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrilu02_analysisNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsrilu02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseScsrilu02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrilu02Native(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsrilu02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrilu02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrilu02Native(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsrilu02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrilu02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrilu02Native(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsrilu02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrilu02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrilu02Native(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsrilu02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSbsrilu02_numericBoost(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseSbsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseSbsrilu02_numericBoostNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseDbsrilu02_numericBoost(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseDbsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseDbsrilu02_numericBoostNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseCbsrilu02_numericBoost(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseCbsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseCbsrilu02_numericBoostNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseZbsrilu02_numericBoost(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseZbsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseZbsrilu02_numericBoostNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseXbsrilu02_zeroPivot(
        cusparseHandle handle, 
        bsrilu02Info info, 
        Pointer position)
    {
        return checkResult(cusparseXbsrilu02_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXbsrilu02_zeroPivotNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        Pointer position);


    public static int cusparseSbsrilu02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSbsrilu02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseSbsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseDbsrilu02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDbsrilu02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseDbsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseCbsrilu02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCbsrilu02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseCbsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseZbsrilu02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZbsrilu02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseZbsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseSbsrilu02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrilu02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseSbsrilu02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrilu02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrilu02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseDbsrilu02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrilu02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrilu02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseCbsrilu02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrilu02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrilu02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseZbsrilu02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSbsrilu02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrilu02Native(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseSbsrilu02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrilu02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrilu02Native(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseDbsrilu02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrilu02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrilu02Native(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseCbsrilu02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrilu02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrilu02Native(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseZbsrilu02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseXcsric02_zeroPivot(
        cusparseHandle handle, 
        csric02Info info, 
        Pointer position)
    {
        return checkResult(cusparseXcsric02_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXcsric02_zeroPivotNative(
        cusparseHandle handle, 
        csric02Info info, 
        Pointer position);


    public static int cusparseScsric02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseScsric02_bufferSizeNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseScsric02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseDcsric02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsric02_bufferSizeNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseDcsric02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseCcsric02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsric02_bufferSizeNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseCcsric02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseZcsric02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsric02_bufferSizeNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseZcsric02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseScsric02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsric02_analysisNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsric02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsric02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsric02_analysisNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsric02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsric02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsric02_analysisNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsric02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsric02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsric02_analysisNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsric02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseScsric02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsric02Native(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsric02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsric02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsric02Native(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsric02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsric02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsric02Native(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsric02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsric02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsric02Native(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsric02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA_valM, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseXbsric02_zeroPivot(
        cusparseHandle handle, 
        bsric02Info info, 
        Pointer position)
    {
        return checkResult(cusparseXbsric02_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXbsric02_zeroPivotNative(
        cusparseHandle handle, 
        bsric02Info info, 
        Pointer position);


    public static int cusparseSbsric02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSbsric02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseSbsric02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseDbsric02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDbsric02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseDbsric02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseCbsric02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCbsric02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseCbsric02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseZbsric02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZbsric02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseZbsric02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int[] pBufferSizeInBytes);


    public static int cusparseSbsric02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer)
    {
        return checkResult(cusparseSbsric02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer));
    }
    private static native int cusparseSbsric02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer);


    public static int cusparseDbsric02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer)
    {
        return checkResult(cusparseDbsric02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer));
    }
    private static native int cusparseDbsric02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer);


    public static int cusparseCbsric02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer)
    {
        return checkResult(cusparseCbsric02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer));
    }
    private static native int cusparseCbsric02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer);


    public static int cusparseZbsric02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer)
    {
        return checkResult(cusparseZbsric02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer));
    }
    private static native int cusparseZbsric02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer);


    public static int cusparseSbsric02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsric02Native(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseSbsric02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsric02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsric02Native(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseDbsric02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsric02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsric02Native(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseCbsric02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsric02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsric02Native(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseZbsric02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSgtsv2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseSgtsv2_bufferSizeExtNative(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
    }
    private static native int cusparseSgtsv2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes);


    public static int cusparseDgtsv2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseDgtsv2_bufferSizeExtNative(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
    }
    private static native int cusparseDgtsv2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes);


    public static int cusparseCgtsv2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseCgtsv2_bufferSizeExtNative(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
    }
    private static native int cusparseCgtsv2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes);


    public static int cusparseZgtsv2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseZgtsv2_bufferSizeExtNative(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
    }
    private static native int cusparseZgtsv2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes);


    public static int cusparseSgtsv2(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgtsv2Native(handle, m, n, dl, d, du, B, ldb, pBuffer));
    }
    private static native int cusparseSgtsv2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer);


    public static int cusparseDgtsv2(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgtsv2Native(handle, m, n, dl, d, du, B, ldb, pBuffer));
    }
    private static native int cusparseDgtsv2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer);


    public static int cusparseCgtsv2(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgtsv2Native(handle, m, n, dl, d, du, B, ldb, pBuffer));
    }
    private static native int cusparseCgtsv2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer);


    public static int cusparseZgtsv2(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgtsv2Native(handle, m, n, dl, d, du, B, ldb, pBuffer));
    }
    private static native int cusparseZgtsv2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer);


    public static int cusparseSgtsv2_nopivot_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseSgtsv2_nopivot_bufferSizeExtNative(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
    }
    private static native int cusparseSgtsv2_nopivot_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes);


    public static int cusparseDgtsv2_nopivot_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseDgtsv2_nopivot_bufferSizeExtNative(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
    }
    private static native int cusparseDgtsv2_nopivot_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes);


    public static int cusparseCgtsv2_nopivot_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseCgtsv2_nopivot_bufferSizeExtNative(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
    }
    private static native int cusparseCgtsv2_nopivot_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes);


    public static int cusparseZgtsv2_nopivot_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseZgtsv2_nopivot_bufferSizeExtNative(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
    }
    private static native int cusparseZgtsv2_nopivot_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        long[] bufferSizeInBytes);


    public static int cusparseSgtsv2_nopivot(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgtsv2_nopivotNative(handle, m, n, dl, d, du, B, ldb, pBuffer));
    }
    private static native int cusparseSgtsv2_nopivotNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer);


    public static int cusparseDgtsv2_nopivot(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgtsv2_nopivotNative(handle, m, n, dl, d, du, B, ldb, pBuffer));
    }
    private static native int cusparseDgtsv2_nopivotNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer);


    public static int cusparseCgtsv2_nopivot(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgtsv2_nopivotNative(handle, m, n, dl, d, du, B, ldb, pBuffer));
    }
    private static native int cusparseCgtsv2_nopivotNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer);


    public static int cusparseZgtsv2_nopivot(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgtsv2_nopivotNative(handle, m, n, dl, d, du, B, ldb, pBuffer));
    }
    private static native int cusparseZgtsv2_nopivotNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb, 
        Pointer pBuffer);


    public static int cusparseSgtsv2StridedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseSgtsv2StridedBatch_bufferSizeExtNative(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes));
    }
    private static native int cusparseSgtsv2StridedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        long[] bufferSizeInBytes);


    public static int cusparseDgtsv2StridedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseDgtsv2StridedBatch_bufferSizeExtNative(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes));
    }
    private static native int cusparseDgtsv2StridedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        long[] bufferSizeInBytes);


    public static int cusparseCgtsv2StridedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseCgtsv2StridedBatch_bufferSizeExtNative(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes));
    }
    private static native int cusparseCgtsv2StridedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        long[] bufferSizeInBytes);


    public static int cusparseZgtsv2StridedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        long[] bufferSizeInBytes)
    {
        return checkResult(cusparseZgtsv2StridedBatch_bufferSizeExtNative(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes));
    }
    private static native int cusparseZgtsv2StridedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        long[] bufferSizeInBytes);


    public static int cusparseSgtsv2StridedBatch(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgtsv2StridedBatchNative(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
    }
    private static native int cusparseSgtsv2StridedBatchNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        Pointer pBuffer);


    public static int cusparseDgtsv2StridedBatch(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgtsv2StridedBatchNative(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
    }
    private static native int cusparseDgtsv2StridedBatchNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        Pointer pBuffer);


    public static int cusparseCgtsv2StridedBatch(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgtsv2StridedBatchNative(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
    }
    private static native int cusparseCgtsv2StridedBatchNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        Pointer pBuffer);


    public static int cusparseZgtsv2StridedBatch(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgtsv2StridedBatchNative(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
    }
    private static native int cusparseZgtsv2StridedBatchNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride, 
        Pointer pBuffer);


    public static int cusparseSgtsvInterleavedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSgtsvInterleavedBatch_bufferSizeExtNative(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes));
    }
    private static native int cusparseSgtsvInterleavedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes);


    public static int cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDgtsvInterleavedBatch_bufferSizeExtNative(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes));
    }
    private static native int cusparseDgtsvInterleavedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes);


    public static int cusparseCgtsvInterleavedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCgtsvInterleavedBatch_bufferSizeExtNative(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes));
    }
    private static native int cusparseCgtsvInterleavedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes);


    public static int cusparseZgtsvInterleavedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZgtsvInterleavedBatch_bufferSizeExtNative(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes));
    }
    private static native int cusparseZgtsvInterleavedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes);


    public static int cusparseSgtsvInterleavedBatch(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgtsvInterleavedBatchNative(handle, algo, m, dl, d, du, x, batchCount, pBuffer));
    }
    private static native int cusparseSgtsvInterleavedBatchNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer);


    public static int cusparseDgtsvInterleavedBatch(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgtsvInterleavedBatchNative(handle, algo, m, dl, d, du, x, batchCount, pBuffer));
    }
    private static native int cusparseDgtsvInterleavedBatchNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer);


    public static int cusparseCgtsvInterleavedBatch(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgtsvInterleavedBatchNative(handle, algo, m, dl, d, du, x, batchCount, pBuffer));
    }
    private static native int cusparseCgtsvInterleavedBatchNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer);


    public static int cusparseZgtsvInterleavedBatch(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgtsvInterleavedBatchNative(handle, algo, m, dl, d, du, x, batchCount, pBuffer));
    }
    private static native int cusparseZgtsvInterleavedBatchNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer);


    public static int cusparseSgpsvInterleavedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSgpsvInterleavedBatch_bufferSizeExtNative(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes));
    }
    private static native int cusparseSgpsvInterleavedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes);


    public static int cusparseDgpsvInterleavedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDgpsvInterleavedBatch_bufferSizeExtNative(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes));
    }
    private static native int cusparseDgpsvInterleavedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes);


    public static int cusparseCgpsvInterleavedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCgpsvInterleavedBatch_bufferSizeExtNative(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes));
    }
    private static native int cusparseCgpsvInterleavedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes);


    public static int cusparseZgpsvInterleavedBatch_bufferSizeExt(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZgpsvInterleavedBatch_bufferSizeExtNative(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes));
    }
    private static native int cusparseZgpsvInterleavedBatch_bufferSizeExtNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        long[] pBufferSizeInBytes);


    public static int cusparseSgpsvInterleavedBatch(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgpsvInterleavedBatchNative(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer));
    }
    private static native int cusparseSgpsvInterleavedBatchNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer);


    public static int cusparseDgpsvInterleavedBatch(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgpsvInterleavedBatchNative(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer));
    }
    private static native int cusparseDgpsvInterleavedBatchNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer);


    public static int cusparseCgpsvInterleavedBatch(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgpsvInterleavedBatchNative(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer));
    }
    private static native int cusparseCgpsvInterleavedBatchNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer);


    public static int cusparseZgpsvInterleavedBatch(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgpsvInterleavedBatchNative(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer));
    }
    private static native int cusparseZgpsvInterleavedBatchNative(
        cusparseHandle handle, 
        int algo, 
        int m, 
        Pointer ds, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer dw, 
        Pointer x, 
        int batchCount, 
        Pointer pBuffer);


    //##############################################################################
    //# EXTRA ROUTINES
    //##############################################################################
    @Deprecated
    public static int cusparseCreateCsrgemm2Info(
        csrgemm2Info info)
    {
        return checkResult(cusparseCreateCsrgemm2InfoNative(info));
    }
    private static native int cusparseCreateCsrgemm2InfoNative(
        csrgemm2Info info);


    @Deprecated
    public static int cusparseDestroyCsrgemm2Info(
        csrgemm2Info info)
    {
        return checkResult(cusparseDestroyCsrgemm2InfoNative(info));
    }
    private static native int cusparseDestroyCsrgemm2InfoNative(
        csrgemm2Info info);


    @Deprecated
    public static int cusparseScsrgemm2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        csrgemm2Info info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseScsrgemm2_bufferSizeExtNative(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes));
    }
    private static native int cusparseScsrgemm2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        csrgemm2Info info, 
        long[] pBufferSizeInBytes);


    @Deprecated
    public static int cusparseDcsrgemm2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        csrgemm2Info info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsrgemm2_bufferSizeExtNative(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes));
    }
    private static native int cusparseDcsrgemm2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        csrgemm2Info info, 
        long[] pBufferSizeInBytes);


    @Deprecated
    public static int cusparseCcsrgemm2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        csrgemm2Info info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsrgemm2_bufferSizeExtNative(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes));
    }
    private static native int cusparseCcsrgemm2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        csrgemm2Info info, 
        long[] pBufferSizeInBytes);


    @Deprecated
    public static int cusparseZcsrgemm2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        csrgemm2Info info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsrgemm2_bufferSizeExtNative(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes));
    }
    private static native int cusparseZcsrgemm2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        csrgemm2Info info, 
        long[] pBufferSizeInBytes);


    @Deprecated
    public static int cusparseXcsrgemm2Nnz(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        csrgemm2Info info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXcsrgemm2NnzNative(handle, m, n, k, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer));
    }
    private static native int cusparseXcsrgemm2NnzNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        csrgemm2Info info, 
        Pointer pBuffer);


    @Deprecated
    public static int cusparseScsrgemm2(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedValD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        csrgemm2Info info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrgemm2Native(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer));
    }
    private static native int cusparseScsrgemm2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedValD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        csrgemm2Info info, 
        Pointer pBuffer);


    @Deprecated
    public static int cusparseDcsrgemm2(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedValD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        csrgemm2Info info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrgemm2Native(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer));
    }
    private static native int cusparseDcsrgemm2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedValD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        csrgemm2Info info, 
        Pointer pBuffer);


    @Deprecated
    public static int cusparseCcsrgemm2(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedValD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        csrgemm2Info info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrgemm2Native(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer));
    }
    private static native int cusparseCcsrgemm2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedValD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        csrgemm2Info info, 
        Pointer pBuffer);


    @Deprecated
    public static int cusparseZcsrgemm2(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedValD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        csrgemm2Info info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrgemm2Native(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer));
    }
    private static native int cusparseZcsrgemm2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        int k, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        Pointer beta, 
        cusparseMatDescr descrD, 
        int nnzD, 
        Pointer csrSortedValD, 
        Pointer csrSortedRowPtrD, 
        Pointer csrSortedColIndD, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        csrgemm2Info info, 
        Pointer pBuffer);


    public static int cusparseScsrgeam2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseScsrgeam2_bufferSizeExtNative(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
    }
    private static native int cusparseScsrgeam2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes);


    public static int cusparseDcsrgeam2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsrgeam2_bufferSizeExtNative(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
    }
    private static native int cusparseDcsrgeam2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes);


    public static int cusparseCcsrgeam2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsrgeam2_bufferSizeExtNative(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
    }
    private static native int cusparseCcsrgeam2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes);


    public static int cusparseZcsrgeam2_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsrgeam2_bufferSizeExtNative(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
    }
    private static native int cusparseZcsrgeam2_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes);


    public static int cusparseXcsrgeam2Nnz(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer workspace)
    {
        return checkResult(cusparseXcsrgeam2NnzNative(handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace));
    }
    private static native int cusparseXcsrgeam2NnzNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer workspace);


    public static int cusparseScsrgeam2(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrgeam2Native(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
    }
    private static native int cusparseScsrgeam2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer);


    public static int cusparseDcsrgeam2(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrgeam2Native(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
    }
    private static native int cusparseDcsrgeam2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer);


    public static int cusparseCcsrgeam2(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrgeam2Native(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
    }
    private static native int cusparseCcsrgeam2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer);


    public static int cusparseZcsrgeam2(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrgeam2Native(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
    }
    private static native int cusparseZcsrgeam2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrSortedValB, 
        Pointer csrSortedRowPtrB, 
        Pointer csrSortedColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer);


    //##############################################################################
    //# SPARSE MATRIX REORDERING
    //##############################################################################
    public static int cusparseScsrcolor(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer fractionToColor, 
        Pointer ncolors, 
        Pointer coloring, 
        Pointer reordering, 
        cusparseColorInfo info)
    {
        return checkResult(cusparseScsrcolorNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info));
    }
    private static native int cusparseScsrcolorNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer fractionToColor, 
        Pointer ncolors, 
        Pointer coloring, 
        Pointer reordering, 
        cusparseColorInfo info);


    public static int cusparseDcsrcolor(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer fractionToColor, 
        Pointer ncolors, 
        Pointer coloring, 
        Pointer reordering, 
        cusparseColorInfo info)
    {
        return checkResult(cusparseDcsrcolorNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info));
    }
    private static native int cusparseDcsrcolorNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer fractionToColor, 
        Pointer ncolors, 
        Pointer coloring, 
        Pointer reordering, 
        cusparseColorInfo info);


    public static int cusparseCcsrcolor(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer fractionToColor, 
        Pointer ncolors, 
        Pointer coloring, 
        Pointer reordering, 
        cusparseColorInfo info)
    {
        return checkResult(cusparseCcsrcolorNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info));
    }
    private static native int cusparseCcsrcolorNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer fractionToColor, 
        Pointer ncolors, 
        Pointer coloring, 
        Pointer reordering, 
        cusparseColorInfo info);


    public static int cusparseZcsrcolor(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer fractionToColor, 
        Pointer ncolors, 
        Pointer coloring, 
        Pointer reordering, 
        cusparseColorInfo info)
    {
        return checkResult(cusparseZcsrcolorNative(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info));
    }
    private static native int cusparseZcsrcolorNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer fractionToColor, 
        Pointer ncolors, 
        Pointer coloring, 
        Pointer reordering, 
        cusparseColorInfo info);


    //##############################################################################
    //# SPARSE FORMAT CONVERSION
    //##############################################################################
    public static int cusparseSnnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseSnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
    }
    private static native int cusparseSnnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseDnnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseDnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
    }
    private static native int cusparseDnnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseCnnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseCnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
    }
    private static native int cusparseCnnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseZnnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseZnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
    }
    private static native int cusparseZnnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr);


    //##############################################################################
    //# SPARSE FORMAT CONVERSION #
    //##############################################################################
    public static int cusparseSnnz_compress(
        cusparseHandle handle, 
        int m, 
        cusparseMatDescr descr, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer nnzPerRow, 
        Pointer nnzC, 
        float tol)
    {
        return checkResult(cusparseSnnz_compressNative(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol));
    }
    private static native int cusparseSnnz_compressNative(
        cusparseHandle handle, 
        int m, 
        cusparseMatDescr descr, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer nnzPerRow, 
        Pointer nnzC, 
        float tol);


    public static int cusparseDnnz_compress(
        cusparseHandle handle, 
        int m, 
        cusparseMatDescr descr, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer nnzPerRow, 
        Pointer nnzC, 
        double tol)
    {
        return checkResult(cusparseDnnz_compressNative(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol));
    }
    private static native int cusparseDnnz_compressNative(
        cusparseHandle handle, 
        int m, 
        cusparseMatDescr descr, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer nnzPerRow, 
        Pointer nnzC, 
        double tol);


    public static int cusparseCnnz_compress(
        cusparseHandle handle, 
        int m, 
        cusparseMatDescr descr, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer nnzPerRow, 
        Pointer nnzC, 
        cuComplex tol)
    {
        return checkResult(cusparseCnnz_compressNative(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol));
    }
    private static native int cusparseCnnz_compressNative(
        cusparseHandle handle, 
        int m, 
        cusparseMatDescr descr, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer nnzPerRow, 
        Pointer nnzC, 
        cuComplex tol);


    public static int cusparseZnnz_compress(
        cusparseHandle handle, 
        int m, 
        cusparseMatDescr descr, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer nnzPerRow, 
        Pointer nnzC, 
        cuDoubleComplex tol)
    {
        return checkResult(cusparseZnnz_compressNative(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol));
    }
    private static native int cusparseZnnz_compressNative(
        cusparseHandle handle, 
        int m, 
        cusparseMatDescr descr, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer nnzPerRow, 
        Pointer nnzC, 
        cuDoubleComplex tol);


    public static int cusparseScsr2csr_compress(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedColIndA, 
        Pointer csrSortedRowPtrA, 
        int nnzA, 
        Pointer nnzPerRow, 
        Pointer csrSortedValC, 
        Pointer csrSortedColIndC, 
        Pointer csrSortedRowPtrC, 
        float tol)
    {
        return checkResult(cusparseScsr2csr_compressNative(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol));
    }
    private static native int cusparseScsr2csr_compressNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedColIndA, 
        Pointer csrSortedRowPtrA, 
        int nnzA, 
        Pointer nnzPerRow, 
        Pointer csrSortedValC, 
        Pointer csrSortedColIndC, 
        Pointer csrSortedRowPtrC, 
        float tol);


    public static int cusparseDcsr2csr_compress(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedColIndA, 
        Pointer csrSortedRowPtrA, 
        int nnzA, 
        Pointer nnzPerRow, 
        Pointer csrSortedValC, 
        Pointer csrSortedColIndC, 
        Pointer csrSortedRowPtrC, 
        double tol)
    {
        return checkResult(cusparseDcsr2csr_compressNative(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol));
    }
    private static native int cusparseDcsr2csr_compressNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedColIndA, 
        Pointer csrSortedRowPtrA, 
        int nnzA, 
        Pointer nnzPerRow, 
        Pointer csrSortedValC, 
        Pointer csrSortedColIndC, 
        Pointer csrSortedRowPtrC, 
        double tol);


    public static int cusparseCcsr2csr_compress(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedColIndA, 
        Pointer csrSortedRowPtrA, 
        int nnzA, 
        Pointer nnzPerRow, 
        Pointer csrSortedValC, 
        Pointer csrSortedColIndC, 
        Pointer csrSortedRowPtrC, 
        cuComplex tol)
    {
        return checkResult(cusparseCcsr2csr_compressNative(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol));
    }
    private static native int cusparseCcsr2csr_compressNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedColIndA, 
        Pointer csrSortedRowPtrA, 
        int nnzA, 
        Pointer nnzPerRow, 
        Pointer csrSortedValC, 
        Pointer csrSortedColIndC, 
        Pointer csrSortedRowPtrC, 
        cuComplex tol);


    public static int cusparseZcsr2csr_compress(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedColIndA, 
        Pointer csrSortedRowPtrA, 
        int nnzA, 
        Pointer nnzPerRow, 
        Pointer csrSortedValC, 
        Pointer csrSortedColIndC, 
        Pointer csrSortedRowPtrC, 
        cuDoubleComplex tol)
    {
        return checkResult(cusparseZcsr2csr_compressNative(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol));
    }
    private static native int cusparseZcsr2csr_compressNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedColIndA, 
        Pointer csrSortedRowPtrA, 
        int nnzA, 
        Pointer nnzPerRow, 
        Pointer csrSortedValC, 
        Pointer csrSortedColIndC, 
        Pointer csrSortedRowPtrC, 
        cuDoubleComplex tol);


    @Deprecated
    public static int cusparseSdense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA)
    {
        return checkResult(cusparseSdense2csrNative(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA));
    }
    private static native int cusparseSdense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA);


    @Deprecated
    public static int cusparseDdense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA)
    {
        return checkResult(cusparseDdense2csrNative(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA));
    }
    private static native int cusparseDdense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA);


    @Deprecated
    public static int cusparseCdense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA)
    {
        return checkResult(cusparseCdense2csrNative(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA));
    }
    private static native int cusparseCdense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA);


    @Deprecated
    public static int cusparseZdense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA)
    {
        return checkResult(cusparseZdense2csrNative(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA));
    }
    private static native int cusparseZdense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA);


    @Deprecated
    public static int cusparseScsr2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseScsr2denseNative(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda));
    }
    private static native int cusparseScsr2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer A, 
        int lda);


    @Deprecated
    public static int cusparseDcsr2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseDcsr2denseNative(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda));
    }
    private static native int cusparseDcsr2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer A, 
        int lda);


    @Deprecated
    public static int cusparseCcsr2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseCcsr2denseNative(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda));
    }
    private static native int cusparseCcsr2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer A, 
        int lda);


    @Deprecated
    public static int cusparseZcsr2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseZcsr2denseNative(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda));
    }
    private static native int cusparseZcsr2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer A, 
        int lda);


    @Deprecated
    public static int cusparseSdense2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA)
    {
        return checkResult(cusparseSdense2cscNative(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA));
    }
    private static native int cusparseSdense2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA);


    @Deprecated
    public static int cusparseDdense2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA)
    {
        return checkResult(cusparseDdense2cscNative(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA));
    }
    private static native int cusparseDdense2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA);


    @Deprecated
    public static int cusparseCdense2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA)
    {
        return checkResult(cusparseCdense2cscNative(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA));
    }
    private static native int cusparseCdense2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA);


    @Deprecated
    public static int cusparseZdense2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA)
    {
        return checkResult(cusparseZdense2cscNative(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA));
    }
    private static native int cusparseZdense2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA);


    @Deprecated
    public static int cusparseScsc2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseScsc2denseNative(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda));
    }
    private static native int cusparseScsc2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA, 
        Pointer A, 
        int lda);


    @Deprecated
    public static int cusparseDcsc2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseDcsc2denseNative(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda));
    }
    private static native int cusparseDcsc2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA, 
        Pointer A, 
        int lda);


    @Deprecated
    public static int cusparseCcsc2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseCcsc2denseNative(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda));
    }
    private static native int cusparseCcsc2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA, 
        Pointer A, 
        int lda);


    @Deprecated
    public static int cusparseZcsc2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseZcsc2denseNative(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda));
    }
    private static native int cusparseZcsc2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscSortedValA, 
        Pointer cscSortedRowIndA, 
        Pointer cscSortedColPtrA, 
        Pointer A, 
        int lda);


    public static int cusparseXcoo2csr(
        cusparseHandle handle, 
        Pointer cooRowInd, 
        int nnz, 
        int m, 
        Pointer csrSortedRowPtr, 
        int idxBase)
    {
        return checkResult(cusparseXcoo2csrNative(handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase));
    }
    private static native int cusparseXcoo2csrNative(
        cusparseHandle handle, 
        Pointer cooRowInd, 
        int nnz, 
        int m, 
        Pointer csrSortedRowPtr, 
        int idxBase);


    public static int cusparseXcsr2coo(
        cusparseHandle handle, 
        Pointer csrSortedRowPtr, 
        int nnz, 
        int m, 
        Pointer cooRowInd, 
        int idxBase)
    {
        return checkResult(cusparseXcsr2cooNative(handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase));
    }
    private static native int cusparseXcsr2cooNative(
        cusparseHandle handle, 
        Pointer csrSortedRowPtr, 
        int nnz, 
        int m, 
        Pointer cooRowInd, 
        int idxBase);


    public static int cusparseXcsr2bsrNnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseXcsr2bsrNnzNative(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedRowPtrC, nnzTotalDevHostPtr));
    }
    private static native int cusparseXcsr2bsrNnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseScsr2bsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC)
    {
        return checkResult(cusparseScsr2bsrNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC));
    }
    private static native int cusparseScsr2bsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC);


    public static int cusparseDcsr2bsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC)
    {
        return checkResult(cusparseDcsr2bsrNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC));
    }
    private static native int cusparseDcsr2bsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC);


    public static int cusparseCcsr2bsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC)
    {
        return checkResult(cusparseCcsr2bsrNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC));
    }
    private static native int cusparseCcsr2bsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC);


    public static int cusparseZcsr2bsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC)
    {
        return checkResult(cusparseZcsr2bsrNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC));
    }
    private static native int cusparseZcsr2bsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC);


    public static int cusparseSbsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC)
    {
        return checkResult(cusparseSbsr2csrNative(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC));
    }
    private static native int cusparseSbsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC);


    public static int cusparseDbsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC)
    {
        return checkResult(cusparseDbsr2csrNative(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC));
    }
    private static native int cusparseDbsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC);


    public static int cusparseCbsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC)
    {
        return checkResult(cusparseCbsr2csrNative(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC));
    }
    private static native int cusparseCbsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC);


    public static int cusparseZbsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC)
    {
        return checkResult(cusparseZbsr2csrNative(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC));
    }
    private static native int cusparseZbsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC);


    public static int cusparseSgebsr2gebsc_bufferSize(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSgebsr2gebsc_bufferSizeNative(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseSgebsr2gebsc_bufferSizeNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes);


    public static int cusparseDgebsr2gebsc_bufferSize(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDgebsr2gebsc_bufferSizeNative(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseDgebsr2gebsc_bufferSizeNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes);


    public static int cusparseCgebsr2gebsc_bufferSize(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCgebsr2gebsc_bufferSizeNative(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseCgebsr2gebsc_bufferSizeNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes);


    public static int cusparseZgebsr2gebsc_bufferSize(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZgebsr2gebsc_bufferSizeNative(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseZgebsr2gebsc_bufferSizeNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes);


    public static int cusparseSgebsr2gebsc(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int idxBase, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgebsr2gebscNative(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer));
    }
    private static native int cusparseSgebsr2gebscNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int idxBase, 
        Pointer pBuffer);


    public static int cusparseDgebsr2gebsc(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int idxBase, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgebsr2gebscNative(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer));
    }
    private static native int cusparseDgebsr2gebscNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int idxBase, 
        Pointer pBuffer);


    public static int cusparseCgebsr2gebsc(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int idxBase, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgebsr2gebscNative(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer));
    }
    private static native int cusparseCgebsr2gebscNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int idxBase, 
        Pointer pBuffer);


    public static int cusparseZgebsr2gebsc(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int idxBase, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgebsr2gebscNative(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer));
    }
    private static native int cusparseZgebsr2gebscNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrSortedVal, 
        Pointer bsrSortedRowPtr, 
        Pointer bsrSortedColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int idxBase, 
        Pointer pBuffer);


    public static int cusparseSgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC)
    {
        return checkResult(cusparseSgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC));
    }
    private static native int cusparseSgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC);


    public static int cusparseDgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC)
    {
        return checkResult(cusparseDgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC));
    }
    private static native int cusparseDgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC);


    public static int cusparseCgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC)
    {
        return checkResult(cusparseCgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC));
    }
    private static native int cusparseCgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC);


    public static int cusparseZgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC)
    {
        return checkResult(cusparseZgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC));
    }
    private static native int cusparseZgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC);


    public static int cusparseScsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseScsr2gebsr_bufferSizeNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseScsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes);


    public static int cusparseDcsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsr2gebsr_bufferSizeNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseDcsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes);


    public static int cusparseCcsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsr2gebsr_bufferSizeNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseCcsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes);


    public static int cusparseZcsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsr2gebsr_bufferSizeNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseZcsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        int[] pBufferSizeInBytes);


    public static int cusparseXcsr2gebsrNnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedRowPtrC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXcsr2gebsrNnzNative(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedRowPtrC, rowBlockDim, colBlockDim, nnzTotalDevHostPtr, pBuffer));
    }
    private static native int cusparseXcsr2gebsrNnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedRowPtrC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer);


    public static int cusparseScsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsr2gebsrNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer));
    }
    private static native int cusparseScsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer);


    public static int cusparseDcsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsr2gebsrNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer));
    }
    private static native int cusparseDcsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer);


    public static int cusparseCcsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsr2gebsrNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer));
    }
    private static native int cusparseCcsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer);


    public static int cusparseZcsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsr2gebsrNative(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer));
    }
    private static native int cusparseZcsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer);


    public static int cusparseSgebsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSgebsr2gebsr_bufferSizeNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes));
    }
    private static native int cusparseSgebsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        int[] pBufferSizeInBytes);


    public static int cusparseDgebsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDgebsr2gebsr_bufferSizeNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes));
    }
    private static native int cusparseDgebsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        int[] pBufferSizeInBytes);


    public static int cusparseCgebsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCgebsr2gebsr_bufferSizeNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes));
    }
    private static native int cusparseCgebsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        int[] pBufferSizeInBytes);


    public static int cusparseZgebsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        int[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZgebsr2gebsr_bufferSizeNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes));
    }
    private static native int cusparseZgebsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        int[] pBufferSizeInBytes);


    public static int cusparseXgebsr2gebsrNnz(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedRowPtrC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXgebsr2gebsrNnzNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, nnzTotalDevHostPtr, pBuffer));
    }
    private static native int cusparseXgebsr2gebsrNnzNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedRowPtrC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer);


    public static int cusparseSgebsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgebsr2gebsrNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer));
    }
    private static native int cusparseSgebsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer);


    public static int cusparseDgebsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgebsr2gebsrNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer));
    }
    private static native int cusparseDgebsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer);


    public static int cusparseCgebsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgebsr2gebsrNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer));
    }
    private static native int cusparseCgebsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer);


    public static int cusparseZgebsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgebsr2gebsrNative(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer));
    }
    private static native int cusparseZgebsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrSortedValA, 
        Pointer bsrSortedRowPtrA, 
        Pointer bsrSortedColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrSortedValC, 
        Pointer bsrSortedRowPtrC, 
        Pointer bsrSortedColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer);


    //##############################################################################
    //# SPARSE MATRIX SORTING
    //##############################################################################
    public static int cusparseCreateIdentityPermutation(
        cusparseHandle handle, 
        int n, 
        Pointer p)
    {
        return checkResult(cusparseCreateIdentityPermutationNative(handle, n, p));
    }
    private static native int cusparseCreateIdentityPermutationNative(
        cusparseHandle handle, 
        int n, 
        Pointer p);


    public static int cusparseXcoosort_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer cooRowsA, 
        Pointer cooColsA, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseXcoosort_bufferSizeExtNative(handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes));
    }
    private static native int cusparseXcoosort_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer cooRowsA, 
        Pointer cooColsA, 
        long[] pBufferSizeInBytes);


    public static int cusparseXcoosortByRow(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer cooRowsA, 
        Pointer cooColsA, 
        Pointer P, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXcoosortByRowNative(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer));
    }
    private static native int cusparseXcoosortByRowNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer cooRowsA, 
        Pointer cooColsA, 
        Pointer P, 
        Pointer pBuffer);


    public static int cusparseXcoosortByColumn(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer cooRowsA, 
        Pointer cooColsA, 
        Pointer P, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXcoosortByColumnNative(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer));
    }
    private static native int cusparseXcoosortByColumnNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer cooRowsA, 
        Pointer cooColsA, 
        Pointer P, 
        Pointer pBuffer);


    public static int cusparseXcsrsort_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseXcsrsort_bufferSizeExtNative(handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes));
    }
    private static native int cusparseXcsrsort_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        long[] pBufferSizeInBytes);


    public static int cusparseXcsrsort(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer P, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXcsrsortNative(handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer));
    }
    private static native int cusparseXcsrsortNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer P, 
        Pointer pBuffer);


    public static int cusparseXcscsort_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer cscColPtrA, 
        Pointer cscRowIndA, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseXcscsort_bufferSizeExtNative(handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes));
    }
    private static native int cusparseXcscsort_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer cscColPtrA, 
        Pointer cscRowIndA, 
        long[] pBufferSizeInBytes);


    public static int cusparseXcscsort(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer cscColPtrA, 
        Pointer cscRowIndA, 
        Pointer P, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXcscsortNative(handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P, pBuffer));
    }
    private static native int cusparseXcscsortNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer cscColPtrA, 
        Pointer cscRowIndA, 
        Pointer P, 
        Pointer pBuffer);


    public static int cusparseScsru2csr_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseScsru2csr_bufferSizeExtNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes));
    }
    private static native int cusparseScsru2csr_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        long[] pBufferSizeInBytes);


    public static int cusparseDcsru2csr_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsru2csr_bufferSizeExtNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes));
    }
    private static native int cusparseDcsru2csr_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        long[] pBufferSizeInBytes);


    public static int cusparseCcsru2csr_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsru2csr_bufferSizeExtNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes));
    }
    private static native int cusparseCcsru2csr_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        long[] pBufferSizeInBytes);


    public static int cusparseZcsru2csr_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsru2csr_bufferSizeExtNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes));
    }
    private static native int cusparseZcsru2csr_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        long[] pBufferSizeInBytes);


    public static int cusparseScsru2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsru2csrNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer));
    }
    private static native int cusparseScsru2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer);


    public static int cusparseDcsru2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsru2csrNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer));
    }
    private static native int cusparseDcsru2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer);


    public static int cusparseCcsru2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsru2csrNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer));
    }
    private static native int cusparseCcsru2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer);


    public static int cusparseZcsru2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsru2csrNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer));
    }
    private static native int cusparseZcsru2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer);


    public static int cusparseScsr2csru(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsr2csruNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer));
    }
    private static native int cusparseScsr2csruNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer);


    public static int cusparseDcsr2csru(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsr2csruNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer));
    }
    private static native int cusparseDcsr2csruNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer);


    public static int cusparseCcsr2csru(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsr2csruNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer));
    }
    private static native int cusparseCcsr2csruNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer);


    public static int cusparseZcsr2csru(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsr2csruNative(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer));
    }
    private static native int cusparseZcsr2csruNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        csru2csrInfo info, 
        Pointer pBuffer);


    public static int cusparseSpruneDense2csr_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSpruneDense2csr_bufferSizeExtNative(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
    }
    private static native int cusparseSpruneDense2csr_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes);


    public static int cusparseDpruneDense2csr_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDpruneDense2csr_bufferSizeExtNative(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
    }
    private static native int cusparseDpruneDense2csr_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes);


    public static int cusparseSpruneDense2csrNnz(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSpruneDense2csrNnzNative(handle, m, n, A, lda, threshold, descrC, csrRowPtrC, nnzTotalDevHostPtr, pBuffer));
    }
    private static native int cusparseSpruneDense2csrNnzNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer);


    public static int cusparseDpruneDense2csrNnz(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDpruneDense2csrNnzNative(handle, m, n, A, lda, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer));
    }
    private static native int cusparseDpruneDense2csrNnzNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer);


    public static int cusparseSpruneDense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSpruneDense2csrNative(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
    }
    private static native int cusparseSpruneDense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer);


    public static int cusparseDpruneDense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDpruneDense2csrNative(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
    }
    private static native int cusparseDpruneDense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer);


    public static int cusparseSpruneCsr2csr_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSpruneCsr2csr_bufferSizeExtNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
    }
    private static native int cusparseSpruneCsr2csr_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes);


    public static int cusparseDpruneCsr2csr_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDpruneCsr2csr_bufferSizeExtNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
    }
    private static native int cusparseDpruneCsr2csr_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        long[] pBufferSizeInBytes);


    public static int cusparseSpruneCsr2csrNnz(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSpruneCsr2csrNnzNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer));
    }
    private static native int cusparseSpruneCsr2csrNnzNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer);


    public static int cusparseDpruneCsr2csrNnz(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDpruneCsr2csrNnzNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer));
    }
    private static native int cusparseDpruneCsr2csrNnzNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer);


    public static int cusparseSpruneCsr2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSpruneCsr2csrNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
    }
    private static native int cusparseSpruneCsr2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer);


    public static int cusparseDpruneCsr2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDpruneCsr2csrNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
    }
    private static native int cusparseDpruneCsr2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        Pointer threshold, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        Pointer pBuffer);


    public static int cusparseSpruneDense2csrByPercentage_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSpruneDense2csrByPercentage_bufferSizeExtNative(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes));
    }
    private static native int cusparseSpruneDense2csrByPercentage_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        long[] pBufferSizeInBytes);


    public static int cusparseDpruneDense2csrByPercentage_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDpruneDense2csrByPercentage_bufferSizeExtNative(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes));
    }
    private static native int cusparseDpruneDense2csrByPercentage_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        long[] pBufferSizeInBytes);


    public static int cusparseSpruneDense2csrNnzByPercentage(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        pruneInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSpruneDense2csrNnzByPercentageNative(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer));
    }
    private static native int cusparseSpruneDense2csrNnzByPercentageNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        pruneInfo info, 
        Pointer pBuffer);


    public static int cusparseDpruneDense2csrNnzByPercentage(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        pruneInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDpruneDense2csrNnzByPercentageNative(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer));
    }
    private static native int cusparseDpruneDense2csrNnzByPercentageNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        pruneInfo info, 
        Pointer pBuffer);


    public static int cusparseSpruneDense2csrByPercentage(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSpruneDense2csrByPercentageNative(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer));
    }
    private static native int cusparseSpruneDense2csrByPercentageNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        Pointer pBuffer);


    public static int cusparseDpruneDense2csrByPercentage(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDpruneDense2csrByPercentageNative(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer));
    }
    private static native int cusparseDpruneDense2csrByPercentageNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer A, 
        int lda, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        Pointer pBuffer);


    public static int cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseSpruneCsr2csrByPercentage_bufferSizeExtNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes));
    }
    private static native int cusparseSpruneCsr2csrByPercentage_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        long[] pBufferSizeInBytes);


    public static int cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        long[] pBufferSizeInBytes)
    {
        return checkResult(cusparseDpruneCsr2csrByPercentage_bufferSizeExtNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes));
    }
    private static native int cusparseDpruneCsr2csrByPercentage_bufferSizeExtNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        long[] pBufferSizeInBytes);


    public static int cusparseSpruneCsr2csrNnzByPercentage(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        pruneInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSpruneCsr2csrNnzByPercentageNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer));
    }
    private static native int cusparseSpruneCsr2csrNnzByPercentageNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        pruneInfo info, 
        Pointer pBuffer);


    public static int cusparseDpruneCsr2csrNnzByPercentage(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        pruneInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDpruneCsr2csrNnzByPercentageNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer));
    }
    private static native int cusparseDpruneCsr2csrNnzByPercentageNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedRowPtrC, 
        Pointer nnzTotalDevHostPtr, 
        pruneInfo info, 
        Pointer pBuffer);


    public static int cusparseSpruneCsr2csrByPercentage(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSpruneCsr2csrByPercentageNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer));
    }
    private static native int cusparseSpruneCsr2csrByPercentageNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        Pointer pBuffer);


    public static int cusparseDpruneCsr2csrByPercentage(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDpruneCsr2csrByPercentageNative(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer));
    }
    private static native int cusparseDpruneCsr2csrByPercentageNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnzA, 
        cusparseMatDescr descrA, 
        Pointer csrSortedValA, 
        Pointer csrSortedRowPtrA, 
        Pointer csrSortedColIndA, 
        float percentage, 
        cusparseMatDescr descrC, 
        Pointer csrSortedValC, 
        Pointer csrSortedRowPtrC, 
        Pointer csrSortedColIndC, 
        pruneInfo info, 
        Pointer pBuffer);


    public static int cusparseCsr2cscEx2(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscColPtr, 
        Pointer cscRowInd, 
        int valType, 
        int copyValues, 
        int idxBase, 
        int alg, 
        Pointer buffer)
    {
        return checkResult(cusparseCsr2cscEx2Native(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer));
    }
    private static native int cusparseCsr2cscEx2Native(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscColPtr, 
        Pointer cscRowInd, 
        int valType, 
        int copyValues, 
        int idxBase, 
        int alg, 
        Pointer buffer);


    public static int cusparseCsr2cscEx2_bufferSize(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscColPtr, 
        Pointer cscRowInd, 
        int valType, 
        int copyValues, 
        int idxBase, 
        int alg, 
        long[] bufferSize)
    {
        return checkResult(cusparseCsr2cscEx2_bufferSizeNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, bufferSize));
    }
    private static native int cusparseCsr2cscEx2_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscColPtr, 
        Pointer cscRowInd, 
        int valType, 
        int copyValues, 
        int idxBase, 
        int alg, 
        long[] bufferSize);


    // #############################################################################
    // # SPARSE VECTOR DESCRIPTOR
    // #############################################################################
    public static int cusparseCreateSpVec(
        cusparseSpVecDescr spVecDescr, 
        long size, 
        long nnz, 
        Pointer indices, 
        Pointer values, 
        int idxType, 
        int idxBase, 
        int valueType)
    {
        return checkResult(cusparseCreateSpVecNative(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType));
    }
    private static native int cusparseCreateSpVecNative(
        cusparseSpVecDescr spVecDescr, 
        long size, 
        long nnz, 
        Pointer indices, 
        Pointer values, 
        int idxType, 
        int idxBase, 
        int valueType);


    public static int cusparseDestroySpVec(
        cusparseSpVecDescr spVecDescr)
    {
        return checkResult(cusparseDestroySpVecNative(spVecDescr));
    }
    private static native int cusparseDestroySpVecNative(
        cusparseSpVecDescr spVecDescr);


    public static int cusparseSpVecGet(
        cusparseSpVecDescr spVecDescr, 
        long[] size, 
        long[] nnz, 
        Pointer indices, 
        Pointer values, 
        int[] idxType, 
        int[] idxBase, 
        int[] valueType)
    {
        return checkResult(cusparseSpVecGetNative(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType));
    }
    private static native int cusparseSpVecGetNative(
        cusparseSpVecDescr spVecDescr, 
        long[] size, 
        long[] nnz, 
        Pointer indices, 
        Pointer values, 
        int[] idxType, 
        int[] idxBase, 
        int[] valueType);


    public static int cusparseSpVecGetIndexBase(
        cusparseSpVecDescr spVecDescr, 
        int[] idxBase)
    {
        return checkResult(cusparseSpVecGetIndexBaseNative(spVecDescr, idxBase));
    }
    private static native int cusparseSpVecGetIndexBaseNative(
        cusparseSpVecDescr spVecDescr, 
        int[] idxBase);


    public static int cusparseSpVecGetValues(
        cusparseSpVecDescr spVecDescr, 
        Pointer values)
    {
        return checkResult(cusparseSpVecGetValuesNative(spVecDescr, values));
    }
    private static native int cusparseSpVecGetValuesNative(
        cusparseSpVecDescr spVecDescr, 
        Pointer values);


    public static int cusparseSpVecSetValues(
        cusparseSpVecDescr spVecDescr, 
        Pointer values)
    {
        return checkResult(cusparseSpVecSetValuesNative(spVecDescr, values));
    }
    private static native int cusparseSpVecSetValuesNative(
        cusparseSpVecDescr spVecDescr, 
        Pointer values);


    // #############################################################################
    // # DENSE VECTOR DESCRIPTOR
    // #############################################################################
    public static int cusparseCreateDnVec(
        cusparseDnVecDescr dnVecDescr, 
        long size, 
        Pointer values, 
        int valueType)
    {
        return checkResult(cusparseCreateDnVecNative(dnVecDescr, size, values, valueType));
    }
    private static native int cusparseCreateDnVecNative(
        cusparseDnVecDescr dnVecDescr, 
        long size, 
        Pointer values, 
        int valueType);


    public static int cusparseDestroyDnVec(
        cusparseDnVecDescr dnVecDescr)
    {
        return checkResult(cusparseDestroyDnVecNative(dnVecDescr));
    }
    private static native int cusparseDestroyDnVecNative(
        cusparseDnVecDescr dnVecDescr);


    public static int cusparseDnVecGet(
        cusparseDnVecDescr dnVecDescr, 
        long[] size, 
        Pointer values, 
        int[] valueType)
    {
        return checkResult(cusparseDnVecGetNative(dnVecDescr, size, values, valueType));
    }
    private static native int cusparseDnVecGetNative(
        cusparseDnVecDescr dnVecDescr, 
        long[] size, 
        Pointer values, 
        int[] valueType);


    public static int cusparseDnVecGetValues(
        cusparseDnVecDescr dnVecDescr, 
        Pointer values)
    {
        return checkResult(cusparseDnVecGetValuesNative(dnVecDescr, values));
    }
    private static native int cusparseDnVecGetValuesNative(
        cusparseDnVecDescr dnVecDescr, 
        Pointer values);


    public static int cusparseDnVecSetValues(
        cusparseDnVecDescr dnVecDescr, 
        Pointer values)
    {
        return checkResult(cusparseDnVecSetValuesNative(dnVecDescr, values));
    }
    private static native int cusparseDnVecSetValuesNative(
        cusparseDnVecDescr dnVecDescr, 
        Pointer values);


    // #############################################################################
    // # SPARSE MATRIX DESCRIPTOR
    // #############################################################################
    public static int cusparseDestroySpMat(
        cusparseSpMatDescr spMatDescr)
    {
        return checkResult(cusparseDestroySpMatNative(spMatDescr));
    }
    private static native int cusparseDestroySpMatNative(
        cusparseSpMatDescr spMatDescr);


    public static int cusparseSpMatGetFormat(
        cusparseSpMatDescr spMatDescr, 
        int[] format)
    {
        return checkResult(cusparseSpMatGetFormatNative(spMatDescr, format));
    }
    private static native int cusparseSpMatGetFormatNative(
        cusparseSpMatDescr spMatDescr, 
        int[] format);


    public static int cusparseSpMatGetIndexBase(
        cusparseSpMatDescr spMatDescr, 
        int[] idxBase)
    {
        return checkResult(cusparseSpMatGetIndexBaseNative(spMatDescr, idxBase));
    }
    private static native int cusparseSpMatGetIndexBaseNative(
        cusparseSpMatDescr spMatDescr, 
        int[] idxBase);


    public static int cusparseSpMatGetValues(
        cusparseSpMatDescr spMatDescr, 
        Pointer values)
    {
        return checkResult(cusparseSpMatGetValuesNative(spMatDescr, values));
    }
    private static native int cusparseSpMatGetValuesNative(
        cusparseSpMatDescr spMatDescr, 
        Pointer values);


    public static int cusparseSpMatSetValues(
        cusparseSpMatDescr spMatDescr, 
        Pointer values)
    {
        return checkResult(cusparseSpMatSetValuesNative(spMatDescr, values));
    }
    private static native int cusparseSpMatSetValuesNative(
        cusparseSpMatDescr spMatDescr, 
        Pointer values);


    public static int cusparseSpMatGetSize(
        cusparseSpMatDescr spMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] nnz)
    {
        return checkResult(cusparseSpMatGetSizeNative(spMatDescr, rows, cols, nnz));
    }
    private static native int cusparseSpMatGetSizeNative(
        cusparseSpMatDescr spMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] nnz);


    public static int cusparseSpMatSetStridedBatch(
        cusparseSpMatDescr spMatDescr, 
        int batchCount)
    {
        return checkResult(cusparseSpMatSetStridedBatchNative(spMatDescr, batchCount));
    }
    private static native int cusparseSpMatSetStridedBatchNative(
        cusparseSpMatDescr spMatDescr, 
        int batchCount);


    public static int cusparseSpMatGetStridedBatch(
        cusparseSpMatDescr spMatDescr, 
        int[] batchCount)
    {
        return checkResult(cusparseSpMatGetStridedBatchNative(spMatDescr, batchCount));
    }
    private static native int cusparseSpMatGetStridedBatchNative(
        cusparseSpMatDescr spMatDescr, 
        int[] batchCount);


    public static int cusparseCooSetStridedBatch(
        cusparseSpMatDescr spMatDescr, 
        int batchCount, 
        long batchStride)
    {
        return checkResult(cusparseCooSetStridedBatchNative(spMatDescr, batchCount, batchStride));
    }
    private static native int cusparseCooSetStridedBatchNative(
        cusparseSpMatDescr spMatDescr, 
        int batchCount, 
        long batchStride);


    public static int cusparseCsrSetStridedBatch(
        cusparseSpMatDescr spMatDescr, 
        int batchCount, 
        long offsetsBatchStride, 
        long columnsValuesBatchStride)
    {
        return checkResult(cusparseCsrSetStridedBatchNative(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride));
    }
    private static native int cusparseCsrSetStridedBatchNative(
        cusparseSpMatDescr spMatDescr, 
        int batchCount, 
        long offsetsBatchStride, 
        long columnsValuesBatchStride);


    //------------------------------------------------------------------------------
    // ### CSR ###
    public static int cusparseCreateCsr(
        cusparseSpMatDescr spMatDescr, 
        long rows, 
        long cols, 
        long nnz, 
        Pointer csrRowOffsets, 
        Pointer csrColInd, 
        Pointer csrValues, 
        int csrRowOffsetsType, 
        int csrColIndType, 
        int idxBase, 
        int valueType)
    {
        return checkResult(cusparseCreateCsrNative(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType));
    }
    private static native int cusparseCreateCsrNative(
        cusparseSpMatDescr spMatDescr, 
        long rows, 
        long cols, 
        long nnz, 
        Pointer csrRowOffsets, 
        Pointer csrColInd, 
        Pointer csrValues, 
        int csrRowOffsetsType, 
        int csrColIndType, 
        int idxBase, 
        int valueType);


    public static int cusparseCreateCsc(
        cusparseSpMatDescr spMatDescr, 
        long rows, 
        long cols, 
        long nnz, 
        Pointer csrColOffsets, 
        Pointer csrRowInd, 
        Pointer csrValues, 
        int csrColOffsetsType, 
        int csrRowIndType, 
        int idxBase, 
        int valueType)
    {
        return checkResult(cusparseCreateCscNative(spMatDescr, rows, cols, nnz, csrColOffsets, csrRowInd, csrValues, csrColOffsetsType, csrRowIndType, idxBase, valueType));
    }
    private static native int cusparseCreateCscNative(
        cusparseSpMatDescr spMatDescr, 
        long rows, 
        long cols, 
        long nnz, 
        Pointer csrColOffsets, 
        Pointer csrRowInd, 
        Pointer csrValues, 
        int csrColOffsetsType, 
        int csrRowIndType, 
        int idxBase, 
        int valueType);


    public static int cusparseCsrGet(
        cusparseSpMatDescr spMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] nnz, 
        Pointer csrRowOffsets, 
        Pointer csrColInd, 
        Pointer csrValues, 
        int[] csrRowOffsetsType, 
        int[] csrColIndType, 
        int[] idxBase, 
        int[] valueType)
    {
        return checkResult(cusparseCsrGetNative(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType));
    }
    private static native int cusparseCsrGetNative(
        cusparseSpMatDescr spMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] nnz, 
        Pointer csrRowOffsets, 
        Pointer csrColInd, 
        Pointer csrValues, 
        int[] csrRowOffsetsType, 
        int[] csrColIndType, 
        int[] idxBase, 
        int[] valueType);


    public static int cusparseCsrSetPointers(
        cusparseSpMatDescr spMatDescr, 
        Pointer csrRowOffsets, 
        Pointer csrColInd, 
        Pointer csrValues)
    {
        return checkResult(cusparseCsrSetPointersNative(spMatDescr, csrRowOffsets, csrColInd, csrValues));
    }
    private static native int cusparseCsrSetPointersNative(
        cusparseSpMatDescr spMatDescr, 
        Pointer csrRowOffsets, 
        Pointer csrColInd, 
        Pointer csrValues);


    public static int cusparseCscSetPointers(
        cusparseSpMatDescr spMatDescr, 
        Pointer cscColOffsets, 
        Pointer cscRowInd, 
        Pointer cscValues)
    {
        return checkResult(cusparseCscSetPointersNative(spMatDescr, cscColOffsets, cscRowInd, cscValues));
    }
    private static native int cusparseCscSetPointersNative(
        cusparseSpMatDescr spMatDescr, 
        Pointer cscColOffsets, 
        Pointer cscRowInd, 
        Pointer cscValues);


    //------------------------------------------------------------------------------
    // ### COO ###
    public static int cusparseCreateCoo(
        cusparseSpMatDescr spMatDescr, 
        long rows, 
        long cols, 
        long nnz, 
        Pointer cooRowInd, 
        Pointer cooColInd, 
        Pointer cooValues, 
        int cooIdxType, 
        int idxBase, 
        int valueType)
    {
        return checkResult(cusparseCreateCooNative(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType));
    }
    private static native int cusparseCreateCooNative(
        cusparseSpMatDescr spMatDescr, 
        long rows, 
        long cols, 
        long nnz, 
        Pointer cooRowInd, 
        Pointer cooColInd, 
        Pointer cooValues, 
        int cooIdxType, 
        int idxBase, 
        int valueType);


    public static int cusparseCreateCooAoS(
        cusparseSpMatDescr spMatDescr, 
        long rows, 
        long cols, 
        long nnz, 
        Pointer cooInd, 
        Pointer cooValues, 
        int cooIdxType, 
        int idxBase, 
        int valueType)
    {
        return checkResult(cusparseCreateCooAoSNative(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, valueType));
    }
    private static native int cusparseCreateCooAoSNative(
        cusparseSpMatDescr spMatDescr, 
        long rows, 
        long cols, 
        long nnz, 
        Pointer cooInd, 
        Pointer cooValues, 
        int cooIdxType, 
        int idxBase, 
        int valueType);


    public static int cusparseCooGet(
        cusparseSpMatDescr spMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] nnz, 
        Pointer cooRowInd, // COO row indices
        Pointer cooColInd, // COO column indices
        Pointer cooValues, // COO values
        int[] idxType, 
        int[] idxBase, 
        int[] valueType)
    {
        return checkResult(cusparseCooGetNative(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType));
    }
    private static native int cusparseCooGetNative(
        cusparseSpMatDescr spMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] nnz, 
        Pointer cooRowInd, // COO row indices
        Pointer cooColInd, // COO column indices
        Pointer cooValues, // COO values
        int[] idxType, 
        int[] idxBase, 
        int[] valueType);


    public static int cusparseCooAoSGet(
        cusparseSpMatDescr spMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] nnz, 
        Pointer cooInd, // COO indices
        Pointer cooValues, // COO values
        int[] idxType, 
        int[] idxBase, 
        int[] valueType)
    {
        return checkResult(cusparseCooAoSGetNative(spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType, idxBase, valueType));
    }
    private static native int cusparseCooAoSGetNative(
        cusparseSpMatDescr spMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] nnz, 
        Pointer cooInd, // COO indices
        Pointer cooValues, // COO values
        int[] idxType, 
        int[] idxBase, 
        int[] valueType);


    public static int cusparseCooSetPointers(
        cusparseSpMatDescr spMatDescr, 
        Pointer cooRows, 
        Pointer cooColumns, 
        Pointer cooValues)
    {
        return checkResult(cusparseCooSetPointersNative(spMatDescr, cooRows, cooColumns, cooValues));
    }
    private static native int cusparseCooSetPointersNative(
        cusparseSpMatDescr spMatDescr, 
        Pointer cooRows, 
        Pointer cooColumns, 
        Pointer cooValues);


    // #############################################################################
    // # DENSE MATRIX DESCRIPTOR
    // #############################################################################
    public static int cusparseCreateDnMat(
        cusparseDnMatDescr dnMatDescr, 
        long rows, 
        long cols, 
        long ld, 
        Pointer values, 
        int valueType, 
        int order)
    {
        return checkResult(cusparseCreateDnMatNative(dnMatDescr, rows, cols, ld, values, valueType, order));
    }
    private static native int cusparseCreateDnMatNative(
        cusparseDnMatDescr dnMatDescr, 
        long rows, 
        long cols, 
        long ld, 
        Pointer values, 
        int valueType, 
        int order);


    public static int cusparseDestroyDnMat(
        cusparseDnMatDescr dnMatDescr)
    {
        return checkResult(cusparseDestroyDnMatNative(dnMatDescr));
    }
    private static native int cusparseDestroyDnMatNative(
        cusparseDnMatDescr dnMatDescr);


    public static int cusparseDnMatGet(
        cusparseDnMatDescr dnMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] ld, 
        Pointer values, 
        int[] type, 
        int[] order)
    {
        return checkResult(cusparseDnMatGetNative(dnMatDescr, rows, cols, ld, values, type, order));
    }
    private static native int cusparseDnMatGetNative(
        cusparseDnMatDescr dnMatDescr, 
        long[] rows, 
        long[] cols, 
        long[] ld, 
        Pointer values, 
        int[] type, 
        int[] order);


    public static int cusparseDnMatGetValues(
        cusparseDnMatDescr dnMatDescr, 
        Pointer values)
    {
        return checkResult(cusparseDnMatGetValuesNative(dnMatDescr, values));
    }
    private static native int cusparseDnMatGetValuesNative(
        cusparseDnMatDescr dnMatDescr, 
        Pointer values);


    public static int cusparseDnMatSetValues(
        cusparseDnMatDescr dnMatDescr, 
        Pointer values)
    {
        return checkResult(cusparseDnMatSetValuesNative(dnMatDescr, values));
    }
    private static native int cusparseDnMatSetValuesNative(
        cusparseDnMatDescr dnMatDescr, 
        Pointer values);


    public static int cusparseDnMatSetStridedBatch(
        cusparseDnMatDescr dnMatDescr, 
        int batchCount, 
        long batchStride)
    {
        return checkResult(cusparseDnMatSetStridedBatchNative(dnMatDescr, batchCount, batchStride));
    }
    private static native int cusparseDnMatSetStridedBatchNative(
        cusparseDnMatDescr dnMatDescr, 
        int batchCount, 
        long batchStride);


    public static int cusparseDnMatGetStridedBatch(
        cusparseDnMatDescr dnMatDescr, 
        int[] batchCount, 
        long[] batchStride)
    {
        return checkResult(cusparseDnMatGetStridedBatchNative(dnMatDescr, batchCount, batchStride));
    }
    private static native int cusparseDnMatGetStridedBatchNative(
        cusparseDnMatDescr dnMatDescr, 
        int[] batchCount, 
        long[] batchStride);


    // #############################################################################
    // # VECTOR-VECTOR OPERATIONS
    // #############################################################################
    public static int cusparseAxpby(
        cusparseHandle handle, 
        Pointer alpha, 
        cusparseSpVecDescr vecX, 
        Pointer beta, 
        cusparseDnVecDescr vecY)
    {
        return checkResult(cusparseAxpbyNative(handle, alpha, vecX, beta, vecY));
    }
    private static native int cusparseAxpbyNative(
        cusparseHandle handle, 
        Pointer alpha, 
        cusparseSpVecDescr vecX, 
        Pointer beta, 
        cusparseDnVecDescr vecY);


    public static int cusparseGather(
        cusparseHandle handle, 
        cusparseDnVecDescr vecY, 
        cusparseSpVecDescr vecX)
    {
        return checkResult(cusparseGatherNative(handle, vecY, vecX));
    }
    private static native int cusparseGatherNative(
        cusparseHandle handle, 
        cusparseDnVecDescr vecY, 
        cusparseSpVecDescr vecX);


    public static int cusparseScatter(
        cusparseHandle handle, 
        cusparseSpVecDescr vecX, 
        cusparseDnVecDescr vecY)
    {
        return checkResult(cusparseScatterNative(handle, vecX, vecY));
    }
    private static native int cusparseScatterNative(
        cusparseHandle handle, 
        cusparseSpVecDescr vecX, 
        cusparseDnVecDescr vecY);


    public static int cusparseRot(
        cusparseHandle handle, 
        Pointer c_coeff, 
        Pointer s_coeff, 
        cusparseSpVecDescr vecX, 
        cusparseDnVecDescr vecY)
    {
        return checkResult(cusparseRotNative(handle, c_coeff, s_coeff, vecX, vecY));
    }
    private static native int cusparseRotNative(
        cusparseHandle handle, 
        Pointer c_coeff, 
        Pointer s_coeff, 
        cusparseSpVecDescr vecX, 
        cusparseDnVecDescr vecY);


    public static int cusparseSpVV_bufferSize(
        cusparseHandle handle, 
        int opX, 
        cusparseSpVecDescr vecX, 
        cusparseDnVecDescr vecY, 
        Pointer result, 
        int computeType, 
        long[] bufferSize)
    {
        return checkResult(cusparseSpVV_bufferSizeNative(handle, opX, vecX, vecY, result, computeType, bufferSize));
    }
    private static native int cusparseSpVV_bufferSizeNative(
        cusparseHandle handle, 
        int opX, 
        cusparseSpVecDescr vecX, 
        cusparseDnVecDescr vecY, 
        Pointer result, 
        int computeType, 
        long[] bufferSize);


    public static int cusparseSpVV(
        cusparseHandle handle, 
        int opX, 
        cusparseSpVecDescr vecX, 
        cusparseDnVecDescr vecY, 
        Pointer result, 
        int computeType, 
        Pointer externalBuffer)
    {
        return checkResult(cusparseSpVVNative(handle, opX, vecX, vecY, result, computeType, externalBuffer));
    }
    private static native int cusparseSpVVNative(
        cusparseHandle handle, 
        int opX, 
        cusparseSpVecDescr vecX, 
        cusparseDnVecDescr vecY, 
        Pointer result, 
        int computeType, 
        Pointer externalBuffer);


    public static int cusparseSparseToDense_bufferSize(
        cusparseHandle handle, 
        cusparseSpMatDescr matA, 
        cusparseDnMatDescr matB, 
        int alg, 
        long[] bufferSize)
    {
        return checkResult(cusparseSparseToDense_bufferSizeNative(handle, matA, matB, alg, bufferSize));
    }
    private static native int cusparseSparseToDense_bufferSizeNative(
        cusparseHandle handle, 
        cusparseSpMatDescr matA, 
        cusparseDnMatDescr matB, 
        int alg, 
        long[] bufferSize);


    public static int cusparseSparseToDense(
        cusparseHandle handle, 
        cusparseSpMatDescr matA, 
        cusparseDnMatDescr matB, 
        int alg, 
        Pointer buffer)
    {
        return checkResult(cusparseSparseToDenseNative(handle, matA, matB, alg, buffer));
    }
    private static native int cusparseSparseToDenseNative(
        cusparseHandle handle, 
        cusparseSpMatDescr matA, 
        cusparseDnMatDescr matB, 
        int alg, 
        Pointer buffer);


    public static int cusparseDenseToSparse_bufferSize(
        cusparseHandle handle, 
        cusparseDnMatDescr matA, 
        cusparseSpMatDescr matB, 
        int alg, 
        long[] bufferSize)
    {
        return checkResult(cusparseDenseToSparse_bufferSizeNative(handle, matA, matB, alg, bufferSize));
    }
    private static native int cusparseDenseToSparse_bufferSizeNative(
        cusparseHandle handle, 
        cusparseDnMatDescr matA, 
        cusparseSpMatDescr matB, 
        int alg, 
        long[] bufferSize);


    public static int cusparseDenseToSparse_analysis(
        cusparseHandle handle, 
        cusparseDnMatDescr matA, 
        cusparseSpMatDescr matB, 
        int alg, 
        Pointer buffer)
    {
        return checkResult(cusparseDenseToSparse_analysisNative(handle, matA, matB, alg, buffer));
    }
    private static native int cusparseDenseToSparse_analysisNative(
        cusparseHandle handle, 
        cusparseDnMatDescr matA, 
        cusparseSpMatDescr matB, 
        int alg, 
        Pointer buffer);


    public static int cusparseDenseToSparse_convert(
        cusparseHandle handle, 
        cusparseDnMatDescr matA, 
        cusparseSpMatDescr matB, 
        int alg, 
        Pointer buffer)
    {
        return checkResult(cusparseDenseToSparse_convertNative(handle, matA, matB, alg, buffer));
    }
    private static native int cusparseDenseToSparse_convertNative(
        cusparseHandle handle, 
        cusparseDnMatDescr matA, 
        cusparseSpMatDescr matB, 
        int alg, 
        Pointer buffer);


    public static int cusparseSpMV(
        cusparseHandle handle, 
        int opA, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseDnVecDescr vecX, 
        Pointer beta, 
        cusparseDnVecDescr vecY, 
        int computeType, 
        int alg, 
        Pointer externalBuffer)
    {
        return checkResult(cusparseSpMVNative(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer));
    }
    private static native int cusparseSpMVNative(
        cusparseHandle handle, 
        int opA, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseDnVecDescr vecX, 
        Pointer beta, 
        cusparseDnVecDescr vecY, 
        int computeType, 
        int alg, 
        Pointer externalBuffer);


    public static int cusparseSpMV_bufferSize(
        cusparseHandle handle, 
        int opA, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseDnVecDescr vecX, 
        Pointer beta, 
        cusparseDnVecDescr vecY, 
        int computeType, 
        int alg, 
        long[] bufferSize)
    {
        return checkResult(cusparseSpMV_bufferSizeNative(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize));
    }
    private static native int cusparseSpMV_bufferSizeNative(
        cusparseHandle handle, 
        int opA, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseDnVecDescr vecX, 
        Pointer beta, 
        cusparseDnVecDescr vecY, 
        int computeType, 
        int alg, 
        long[] bufferSize);


    public static int cusparseSpMM(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseDnMatDescr matB, 
        Pointer beta, 
        cusparseDnMatDescr matC, 
        int computeType, 
        int alg, 
        Pointer externalBuffer)
    {
        return checkResult(cusparseSpMMNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer));
    }
    private static native int cusparseSpMMNative(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseDnMatDescr matB, 
        Pointer beta, 
        cusparseDnMatDescr matC, 
        int computeType, 
        int alg, 
        Pointer externalBuffer);


    public static int cusparseSpMM_bufferSize(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseDnMatDescr matB, 
        Pointer beta, 
        cusparseDnMatDescr matC, 
        int computeType, 
        int alg, 
        long[] bufferSize)
    {
        return checkResult(cusparseSpMM_bufferSizeNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize));
    }
    private static native int cusparseSpMM_bufferSizeNative(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseDnMatDescr matB, 
        Pointer beta, 
        cusparseDnMatDescr matC, 
        int computeType, 
        int alg, 
        long[] bufferSize);


    public static int cusparseSpGEMM_createDescr(
        cusparseSpGEMMDescr descr)
    {
        return checkResult(cusparseSpGEMM_createDescrNative(descr));
    }
    private static native int cusparseSpGEMM_createDescrNative(
        cusparseSpGEMMDescr descr);


    public static int cusparseSpGEMM_destroyDescr(
        cusparseSpGEMMDescr descr)
    {
        return checkResult(cusparseSpGEMM_destroyDescrNative(descr));
    }
    private static native int cusparseSpGEMM_destroyDescrNative(
        cusparseSpGEMMDescr descr);


    public static int cusparseSpGEMM_workEstimation(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseSpMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        int alg, 
        cusparseSpGEMMDescr spgemmDescr, 
        long[] bufferSize1, 
        Pointer externalBuffer1)
    {
        return checkResult(cusparseSpGEMM_workEstimationNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize1, externalBuffer1));
    }
    private static native int cusparseSpGEMM_workEstimationNative(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseSpMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        int alg, 
        cusparseSpGEMMDescr spgemmDescr, 
        long[] bufferSize1, 
        Pointer externalBuffer1);


    public static int cusparseSpGEMM_compute(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseSpMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        int alg, 
        cusparseSpGEMMDescr spgemmDescr, 
        long[] bufferSize2, 
        Pointer externalBuffer2)
    {
        return checkResult(cusparseSpGEMM_computeNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize2, externalBuffer2));
    }
    private static native int cusparseSpGEMM_computeNative(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseSpMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        int alg, 
        cusparseSpGEMMDescr spgemmDescr, 
        long[] bufferSize2, 
        Pointer externalBuffer2);


    public static int cusparseSpGEMM_copy(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseSpMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        int alg, 
        cusparseSpGEMMDescr spgemmDescr)
    {
        return checkResult(cusparseSpGEMM_copyNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr));
    }
    private static native int cusparseSpGEMM_copyNative(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseSpMatDescr matA, 
        cusparseSpMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        int alg, 
        cusparseSpGEMMDescr spgemmDescr);


    // #############################################################################
    // # GENERAL MATRIX-MATRIX PATTERN-CONSTRAINED MULTIPLICATION
    // #############################################################################
    public static int cusparseConstrainedGeMM(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseDnMatDescr matA, 
        cusparseDnMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        Pointer externalBuffer)
    {
        return checkResult(cusparseConstrainedGeMMNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, externalBuffer));
    }
    private static native int cusparseConstrainedGeMMNative(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseDnMatDescr matA, 
        cusparseDnMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        Pointer externalBuffer);


    public static int cusparseConstrainedGeMM_bufferSize(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseDnMatDescr matA, 
        cusparseDnMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        long[] bufferSize)
    {
        return checkResult(cusparseConstrainedGeMM_bufferSizeNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, bufferSize));
    }
    private static native int cusparseConstrainedGeMM_bufferSizeNative(
        cusparseHandle handle, 
        int opA, 
        int opB, 
        Pointer alpha, 
        cusparseDnMatDescr matA, 
        cusparseDnMatDescr matB, 
        Pointer beta, 
        cusparseSpMatDescr matC, 
        int computeType, 
        long[] bufferSize);


}
