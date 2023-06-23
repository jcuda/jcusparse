/*
 * JCusparse - Java bindings for CUSPARSE, the NVIDIA CUDA sparse
 * matrix library, to be used with JCuda
 *
 * Copyright (c) 2010-2020 Marco Hutter - http://www.jcuda.org
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

public class cusparseSpMMAlg
{
    public static final int CUSPARSE_SPMM_ALG_DEFAULT = 0;
    public static final int CUSPARSE_SPMM_COO_ALG1 = 1;
    public static final int CUSPARSE_SPMM_COO_ALG2 = 2;
    public static final int CUSPARSE_SPMM_COO_ALG3 = 3;
    public static final int CUSPARSE_SPMM_COO_ALG4 = 5;
    public static final int CUSPARSE_SPMM_CSR_ALG1 = 4;
    public static final int CUSPARSE_SPMM_CSR_ALG2 = 6;
    public static final int CUSPARSE_SPMM_CSR_ALG3 = 12;
    public static final int CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseSpMMAlg()
    {
        // Private constructor to prevent instantiation
    }

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_SPMM_ALG_DEFAULT: return "CUSPARSE_SPMM_ALG_DEFAULT";
            case CUSPARSE_SPMM_COO_ALG1: return "CUSPARSE_SPMM_COO_ALG1";
            case CUSPARSE_SPMM_COO_ALG2: return "CUSPARSE_SPMM_COO_ALG2";
            case CUSPARSE_SPMM_COO_ALG3: return "CUSPARSE_SPMM_COO_ALG3";
            case CUSPARSE_SPMM_COO_ALG4: return "CUSPARSE_SPMM_COO_ALG4";
            case CUSPARSE_SPMM_CSR_ALG1: return "CUSPARSE_SPMM_CSR_ALG1";
            case CUSPARSE_SPMM_CSR_ALG2: return "CUSPARSE_SPMM_CSR_ALG2";
            case CUSPARSE_SPMM_CSR_ALG3: return "CUSPARSE_SPMM_CSR_ALG3";
            case CUSPARSE_SPMM_BLOCKED_ELL_ALG1: return "CUSPARSE_SPMM_BLOCKED_ELL_ALG1";
        }
        return "INVALID cusparseSpMMAlg: "+n;
    }
}

