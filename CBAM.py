# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:34:56 2019

@author: dilu

Reference: [2018] CBAM: Convolutional Block Attention Module
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

X = tf.placeholder(tf.float32, shape=[128, 32, 32, 64])

def channel_attention_module(inputs, reduction_ratio, reuse=None, scope='channel_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            
            input_channel = inputs.get_shape().as_list()[-1]
            num_squeeze = input_channel // reduction_ratio
            
            avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            assert avg_pool.get_shape()[1:] == (1, 1, input_channel)
            avg_pool = slim.fully_connected(avg_pool, num_squeeze, activation_fn=None, reuse=None, scope='fc1')
            avg_pool = slim.fully_connected(avg_pool, input_channel, activation_fn=None, reuse=None, scope='fc2')
            assert avg_pool.get_shape()[1:] == (1, 1, input_channel)
            
            max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
            assert max_pool.get_shape()[1:] == (1, 1, input_channel)
            max_pool = slim.fully_connected(max_pool, num_squeeze, activation_fn=None, reuse=True, scope='fc1')
            max_pool = slim.fully_connected(max_pool, input_channel, activation_fn=None, reuse=True, scope='fc2')
            assert max_pool.get_shape()[1:] == (1, 1, input_channel)
            
            scale = tf.nn.sigmoid(avg_pool + max_pool)
            
            channel_attention = scale * inputs
            
            return channel_attention
            
def spatial_attention_module(inputs, kernel_size=7, reuse=None, scope='spatial_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
                            
            avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
            assert max_pool.get_shape()[-1] == 1
            
            concat = tf.concat([avg_pool, max_pool], axis=3)
            assert concat.get_shape()[-1] == 2
            
            concat = slim.conv2d(concat, 1, kernel_size, padding='SAME', activation_fn=None, scope='conv')
            scale = tf.nn.sigmoid(concat)
            
            spatial_attention = scale * inputs
            
            return spatial_attention
            
def cbam_block_channel_first(inputs, reduction_ratio=16, reuse=None, scope='CBAM_Block_Channel_First'):
    with tf.variable_scope(scope, reuse=reuse):
    
        channel_attention = channel_attention_module(inputs, reduction_ratio, reuse=None, scope='channel_attention')
        spatial_attention = spatial_attention_module(channel_attention, kernel_size=7, reuse=None, scope='spatial_attention')
        
        return spatial_attention
        
def cbam_block_spatial_first(inputs, reduction_ratio=16, reuse=None, scope='CBAM_Block_Spatial_First'):
    with tf.variable_scope(scope, reuse=reuse):
        
        spatial_attention = spatial_attention_module(inputs, kernel_size=7, reuse=None, scope='spatial_attention')
        channel_attention = channel_attention_module(spatial_attention, reduction_ratio, reuse=None, scope='channel_attention')
        
        return channel_attention
        
def cbam_block_parallel(inputs, reduction_ratio=16, reuse=None, scope='CBAM_Block_Parallel'):
    with tf.variable_scope(scope, reuse=reuse):
    
        spatial_attention = spatial_attention_module(inputs, kernel_size=7, reuse=None, scope='spatial_attention')
        channel_attention = channel_attention_module(spatial_attention, reduction_ratio, reuse=None, scope='channel_attention')
        
        out = spatial_attention + channel_attention
        
        return out
        

channel_first_output = cbam_block_channel_first(X)
print('Channel first output:', channel_first_output.shape)

spatial_first_output = cbam_block_spatial_first(X)
print('Spatial first output', spatial_first_output.shape)

parallel_output = cbam_block_parallel(X)
print('Parallel output', parallel_output.shape)
