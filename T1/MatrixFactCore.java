package org.petuum.app.matrixfact;

import org.petuum.app.matrixfact.Rating;
import org.petuum.app.matrixfact.LossRecorder;

import org.petuum.ps.PsTableGroup;
import org.petuum.ps.row.double_.DenseDoubleRow;
import org.petuum.ps.row.double_.DenseDoubleRowUpdate;
import org.petuum.ps.row.double_.DoubleRow;
import org.petuum.ps.row.double_.DoubleRowUpdate;
import org.petuum.ps.table.DoubleTable;
import org.petuum.ps.common.util.Timer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class MatrixFactCore {
    private static final Logger logger =
        LoggerFactory.getLogger(MatrixFactCore.class);

    
    // Perform dot-product of LRow and RCol
    public static double dotProduct (DoubleRow rowCache, DoubleRow colCache, int K) {

	double result = 0.0;
	for (int k = 0; k < K; k++) {
		result += rowCache.getUnlocked(k)*colCache.getUnlocked(k);
	} 
	return result;
    }	
    
    public static double delta (DoubleRow rowCache, DoubleRow colCache, int k,  
		double learningRate, double eij, double lambda, double numNonZero) {
	double result = 0.0;
	result = 2*learningRate*(eij*colCache.getUnlocked(k) - (lambda/numNonZero)*rowCache.getUnlocked(k));
	return result;
    } 
    // Perform a single SGD on a rating and update LTable and RTable
    // accordingly.
    public static void sgdOneRating(Rating r, double learningRate,
            DoubleTable LTable, DoubleTable RTable, int K, double lambda) {
        // TODO
	double dij = r.rating;	
	int rowId = r.userId;
	int colId = r.prodId;
	
	// Cache row of LTable and column of RTable
	DoubleRow rowCache = new DenseDoubleRow(K+1);
	DoubleRow lRow = LTable.get(rowId);
	rowCache.reset(lRow);
	
	DoubleRow colCache = new DenseDoubleRow(K+1);
	DoubleRow rCol = RTable.get(colId);
	colCache.reset(rCol);

	// Compute update
	double eij = dij - dotProduct(rowCache, colCache, K);
	double ni = rowCache.getUnlocked(K);
	double mj = colCache.getUnlocked(K);

	DoubleRowUpdate rowUpdates = new DenseDoubleRowUpdate(K+1);
	DoubleRowUpdate colUpdates =  new DenseDoubleRowUpdate(K+1);
	for (int k = 0; k < K; k++) {
		rowUpdates.setUpdate(k, delta(rowCache, colCache, k, learningRate, eij, lambda, ni));
		colUpdates.setUpdate(k, delta(colCache, rowCache, k, learningRate, eij, lambda, mj)); 
	}
	
	// Update row of LTable and column of RTable 
	LTable.batchInc(rowId, rowUpdates);
	RTable.batchInc(colId, colUpdates);
    }

    // Evaluate square loss on entries [elemBegin, elemEnd), and L2-loss on of
    // row [LRowBegin, LRowEnd) of LTable,  [RRowBegin, RRowEnd) of Rtable.
    // Note the interval does not include LRowEnd and RRowEnd. Record the loss to
    // lossRecorder.
    public static void evaluateLoss(ArrayList<Rating> ratings, int ithEval,
            int elemBegin, int elemEnd, DoubleTable LTable,
            DoubleTable RTable, int LRowBegin, int LRowEnd, int RRowBegin,
            int RRowEnd, LossRecorder lossRecorder, int K, double lambda) {
        
	// Compute squared error loss
        double sqLoss = 0.0;
	DoubleRow lRow, rCol;
	DoubleRow rowCache = new DenseDoubleRow(K+1);
	DoubleRow colCache = new DenseDoubleRow(K+1);
		
	for (int r = elemBegin; r < elemEnd; r++) {	
			
	  	// Cache row of LTable and column of RTable
        	lRow = LTable.get(ratings.get(r).userId);
        	rowCache.reset(lRow);
		rCol = RTable.get(ratings.get(r).prodId);
        	colCache.reset(rCol);
		
		sqLoss += Math.pow((double) ratings.get(r).rating - 
				dotProduct(rowCache, colCache, K), 2); 
	}
	
	// Compute Frobenius norm
	double lF = 0.0;
	double rF = 0.0;
	
	for (int i = LRowBegin; i < LRowEnd; i++) {
		// Cache row of LTable
                lRow = LTable.get(i);
                rowCache.reset(lRow);

		for (int k = 0; k < K; k++) {
			lF += Math.pow(rowCache.getUnlocked(k), 2);
		}
	
	}

	 for (int j = RRowBegin; j < RRowEnd; j++) {
                // Cache column of RTable
                rCol = RTable.get(j);
                colCache.reset(rCol);

                for (int k = 0; k < K; k++) {
                        rF += Math.pow(colCache.getUnlocked(k), 2);
                }

        }

        double totalLoss = sqLoss + lambda*(lF + rF);
        lossRecorder.incLoss(ithEval, "SquareLoss", sqLoss);
        lossRecorder.incLoss(ithEval, "FullLoss", totalLoss);
        lossRecorder.incLoss(ithEval, "NumSamples", elemEnd - elemBegin);
    }
}
