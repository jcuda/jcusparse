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

/**
 * Indicates which operations need to be performed with the
 * sparse matrix.
 */
public class cusparseOperation
{
    /**
     * The non-transpose operation is selected.
     */
    public static final int CUSPARSE_OPERATION_NON_TRANSPOSE = 0;

    /**
     * The transpose operation is selected.
     */
    public static final int CUSPARSE_OPERATION_TRANSPOSE = 1;

    /**
     * The conjugate transpose operation is selected.
     */
    public static final int CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseOperation(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_OPERATION_NON_TRANSPOSE: return "CUSPARSE_OPERATION_NON_TRANSPOSE";
            case CUSPARSE_OPERATION_TRANSPOSE: return "CUSPARSE_OPERATION_TRANSPOSE";
            case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE: return "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE";
        }
        return "INVALID cusparseOperation";
    }
}

