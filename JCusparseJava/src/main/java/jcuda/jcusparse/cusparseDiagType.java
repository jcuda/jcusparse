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

/**
 * Indicates if the matrix diagonal entries are unity.
 * The diagonal elements are always assumed to be present, but
 * if CUSPARSE_DIAG_TYPE_UNIT is passed to an API routine,
 * then the routine will assume that all diagonal entries are
 * unity and will not read or modify those entries.
 */
public class cusparseDiagType
{
    /**
     * The matrix diagonal has non-unit elements.
     */
    public static final int CUSPARSE_DIAG_TYPE_NON_UNIT = 0;

    /**
     * The matrix diagonal has unit elements.
     */
    public static final int CUSPARSE_DIAG_TYPE_UNIT = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseDiagType()
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
            case CUSPARSE_DIAG_TYPE_NON_UNIT: return "CUSPARSE_DIAG_TYPE_NON_UNIT";
            case CUSPARSE_DIAG_TYPE_UNIT: return "CUSPARSE_DIAG_TYPE_UNIT";
        }
        return "INVALID cusparseDiagType: "+n;
    }
}

