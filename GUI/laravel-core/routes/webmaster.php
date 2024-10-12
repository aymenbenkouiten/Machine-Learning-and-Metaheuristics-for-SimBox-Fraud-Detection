<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\CDRController;
use App\Http\Controllers\UserController;
use App\Http\Controllers\BarringController;
use App\Http\Controllers\ProfileController;
use App\Http\Controllers\CustomerController;
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\SubscriptionController;


Route::middleware('auth')->group(function () {
    Route::controller(DashboardController::class)->middleware('permission:dashboard_consult')->group(function(){
        Route::get('', "index")->name('dashboard');
        Route::get('documentation', "documentation")->name('documentation');
        Route::get('settings', "settings")->name('settings');
    });

    Route::controller(CustomerController::class)->middleware('permission:customers_consult')->name("customers")->prefix("customers")->group(function(){
        Route::get('', "index");
    });

    Route::controller(SubscriptionController::class)->middleware('permission:subscriptions_consult')->name("subscriptions")->prefix("subscriptions")->group(function(){
        Route::get('', "index");
    });

    Route::controller(CDRController::class)->middleware('permission:cdr_consult')->name("cdr")->prefix("cdr")->group(function(){
        Route::get('', "index");
    });

    Route::controller(BarringController::class)->name("frauds")->prefix("frauds")->group(function(){
        Route::get('suspicious', "suspicious")->name("suspicious")->middleware('permission:suspicious_consult');
        Route::put('suspicious/{suspicious}/fraud', "fraud")->name("fraud")->middleware('permission:suspicious_check');
        Route::put('suspicious/{suspicious}/notfraud', "notfraud")->name("notfraud")->middleware('permission:suspicious_check');

        Route::get('barred', "barred")->name("barred")->middleware('permission:barred_consult');
        Route::put('barred/{barred}/reactive', "reactive")->name("reactive")->middleware('permission:barred_reactive');
    });

    Route::controller(UserController::class)->middleware('permission:users_consult')->name("users")->prefix("users")->group(function(){
        Route::get('', "index");
        Route::get('create', "create")->name("_create")->middleware('permission:users_create');
        Route::post('create', "store")->name("_store")->middleware('permission:users_create');
        Route::get('{user}/edit', "edit")->name("_edit")->middleware('permission:users_edit');
        Route::put('{user}/edit', "update")->name("_update")->middleware('permission:users_edit');
        Route::delete('{user}/delete', "destroy")->name("_delete")->middleware('permission:users_delete');
    });


    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');
});

