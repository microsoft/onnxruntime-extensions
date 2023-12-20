// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef OrtClient_h
#define OrtClient_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface OrtClient : NSObject

+ (BOOL)decodeAndCheckImageWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* OrtClient_h */
