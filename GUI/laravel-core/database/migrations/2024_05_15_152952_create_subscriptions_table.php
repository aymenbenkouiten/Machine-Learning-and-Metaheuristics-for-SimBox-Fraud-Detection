<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('subscriptions', function (Blueprint $table) {
            $table->id();
            $table->string('phone_number')->index();
            $table->string('established_by')->nullable();
            $table->unsignedBigInteger('customer');
            $table->foreign('customer')->references('id')->on('customers');
            $table->unsignedBigInteger('barred_by')->nullable();
            $table->foreign('barred_by')->references('id')->on('users');
            $table->timestamp('barred_at')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('subscriptions');
    }
};
