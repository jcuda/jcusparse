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
 * Indicates the type of matrix stored in sparse storage.
 */
public class cusparseMatrixType
{
    /**
     * The matrix is general.
     */
    public static final int CUSPARSE_MATRIX_TYPE_GENERAL = 0;

    /**
     * The matrix is symmetric.
     */
    public static final int CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1;

    /**
     * The matrix is Hermitian.
     */
    public static final int CUSPARSE_MATRIX_TYPE_HERMITIAN = 2;

    /**
     * The matrix is triangular.
     */
    public static final int CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseMatrixType(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_MATRIX_TYPE_GENERAL: return "CUSPARSE_MATRIX_TYPE_GENERAL";
            case CUSPARSE_MATRIX_TYPE_SYMMETRIC: return "CUSPARSE_MATRIX_TYPE_SYMMETRIC";
            case CUSPARSE_MATRIX_TYPE_HERMITIAN: return "CUSPARSE_MATRIX_TYPE_HERMITIAN";
            case CUSPARSE_MATRIX_TYPE_TRIANGULAR: return "CUSPARSE_MATRIX_TYPE_TRIANGULAR";
        }
        return "INVALID cusparseMatrixType: "+n;
    }
}

