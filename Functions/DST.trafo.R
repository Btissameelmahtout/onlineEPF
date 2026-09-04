DST.trafo<- function(X, Xtime, Xtz= "CET", freq=NULL ){ 
## Xtime in UTC - ONLY for EUROPEAN and NORTH-AMERICAN DATA with change at UTC+1
	Xinit<- as.matrix(X)
	atime.init<- as.numeric(Xtime)
	if(is.null(freq)) freq<- as.numeric(names(which.max(table(diff(atime.init)))))	
	S<- 24*60*60/ freq
	atime<- seq(atime.init[1], atime.init[length(atime.init)], freq)
	idmatch<- match(atime.init, atime)
	X<- array(, dim=c(length(atime), dim(Xinit)[2]))
	X[idmatch,]	<- Xinit


	xx<- as.POSIXct( atime, origin = "1970-01-01", tz=Xtz)
	days<- unique(as.Date(xx, tz=Xtz))   
	if(Xtz %in% c("WET","CET", "EET") ){ # EUROPE
		DST.SPRING<-	grepl("030", format(days, "%m%w")) & as.numeric(format(days, "%d"))>24
		DST.FALL<-	grepl("100", format(days, "%m%w")) & as.numeric(format(days, "%d"))>24
	}
	if(Xtz %in% c("US/Hawaii", "US/Alaska", "US/Pacific", "UC/Mountain", "US/Central", "US/Eastern") ){ # N-America
		DST.SPRING<-	grepl("030", format(days, "%m%w")) & as.numeric(format(days, "%d"))>7 & as.numeric(format(days, "%d"))<15
		DST.FALL<-	grepl("110", format(days, "%m%w")) & as.numeric(format(days, "%d"))<8
	}
	DST<- !(DST.SPRING | DST.FALL)

	xxf.start<- format(xx[1:S], tz=Xtz, usetz=TRUE)
	xxf.end<- format(xx[seq.int(length(xx)-S+1,length(xx))], tz=Xtz, usetz=TRUE)


	DLf<-  format(days) 
	Dlen<- length(DLf)

	Shift<- 2 #for CET
	if(Xtz=="WET") Shift<- 1
	if(Xtz=="EET") Shift<- 3

	Xout<- array(, dim=c(Dlen, S, dim(X)[2]))		
	k<- 0	
## first entry:
	i.d<- 1
	idx<- grep(DLf[i.d], xxf.start)
	if( DST[i.d] ) { 
		Xout[i.d,S+1-rev(1:length(idx)),]<- X[k+1:length(idx),]
	} else {
	#	print(i.d)
		tmp<-(S-rev(idx)+1)
		if(DST.SPRING[i.d]){ 
			## MARCH 
			for(i.S in seq_along(idx) ) {
				if(tmp[i.S]<= Shift* S/24) Xout[i.d,S-S/24- length(idx) + i.S,]<- X[k+i.S,]
				if(tmp[i.S]== Shift* S/24) Xout[i.d,S-S/24- length(idx) + i.S+1:(S/24),]<- t(X[k+i.S,] + t(tcrossprod( (1:(S/24 ) )/(length(1:(S/24))+1), (X[k+i.S+1,]-X[k+i.S,]) )))
				if(tmp[i.S]> Shift* S/24) Xout[i.d,S-S/24- length(idx) + i.S+ S/24,]<- X[k+i.S,]
			}#i.S
		} else { 
			## October
			for(i.S in seq_along(idx) ) {
				if(tmp[i.S]<= Shift* S/24) Xout[i.d,S+S/24- length(idx)+i.S,]<- X[k+i.S,]
				if(tmp[i.S] %in% (Shift*S/24+ 1:(S/24))) Xout[i.d, S+S/24- length(idx)+i.S,]<- 0.5*(X[k+i.S,] + X[k+i.S+S/24,])
				if(tmp[i.S]> (Shift+2)* S/24) Xout[i.d,S+S/24- length(idx)+ i.S-S/24,]<- X[k+i.S,]
			}#i.S
		}
	}# 
	k<- k+ length(idx)
	for(i.d in seq_along(DLf)[c(-1, -length(DLf))] ){ ## first and last extra
		idx<- 1:S
		if( DST[i.d] ) { 
			Xout[i.d,1:length(idx),]<- X[k+1:length(idx),]
		} else {
			if(DST.SPRING[i.d]){ 
				idx<- 1:(S-S/24)
				## MARCH 
				for(i.S in seq_along(idx) ) {
					if(i.S<= Shift* S/24) Xout[i.d,i.S,]<- X[k+i.S,]
					if(i.S== Shift* S/24) Xout[i.d,i.S+1:(S/24),]<- t(X[k+i.S,] + t(tcrossprod( (1:(S/24 ) )/(length(1:(S/24))+1), (X[k+i.S+1,]-X[k+i.S,]) )))
					if(i.S> Shift* S/24) Xout[i.d,i.S+ S/24,]<- X[k+i.S,]
				}#i.S
			} else { 
				idx<- 1:(S+S/24)
				## October
				for(i.S in seq_along(idx) ) {
					if(i.S<= Shift* S/24) Xout[i.d,i.S,]<- X[k+i.S,]
					if(i.S %in% (Shift*S/24+ 1:(S/24))) Xout[i.d, i.S,]<- 0.5*(X[k+i.S,] + X[k+i.S+S/24,])
					if(i.S> (Shift+2)* S/24) Xout[i.d,i.S-S/24,]<- X[k+i.S,]
				}#i.S
			}
		}# 
		k<- k+ length(idx)
	}
## last
	i.d<- length(DLf)
		idx<- grep(DLf[i.d], xxf.end)
		if( DST[i.d] ) { 
			Xout[i.d,1:length(idx),]<- X[k+1:length(idx),]
		} else {
			if(DST.SPRING[i.d]){ 
				## MARCH 
				for(i.S in seq_along(idx) ) {
					if(i.S<= Shift* S/24) Xout[i.d,i.S,]<- X[k+i.S,]
					if(i.S== Shift* S/24) Xout[i.d,i.S+1:(S/24),]<- t(X[k+i.S,] + t(tcrossprod( (1:(S/24 ) )/(length(1:(S/24))+1), (X[k+i.S+1,]-X[k+i.S,]) )))
					if(i.S> Shift* S/24) Xout[i.d,i.S+ S/24,]<- X[k+i.S,]
				}#i.S
			} else { 
				## October
				for(i.S in seq_along(idx) ) {
					if(i.S<= Shift* S/24) Xout[i.d,i.S,]<- X[k+i.S,]
					if(i.S %in% (Shift*S/24+ 1:(S/24))) Xout[i.d, i.S,]<- 0.5*(X[k+i.S,] + X[k+i.S+S/24,])
					if(i.S> (Shift+2)* S/24) Xout[i.d,i.S-S/24,]<- X[k+i.S,]
				}#i.S
			}
		}# 
		k<- k+ length(idx)


	Xout
}

