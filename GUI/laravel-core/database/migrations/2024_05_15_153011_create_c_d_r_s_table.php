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
        Schema::create('c_d_r_s', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('A_Num');
            $table->foreign('A_Num')->references('id')->on('subscriptions');
            $table->integer('Call_Type')->nullable();
            $table->timestamp('Charging_Tm')->nullable();
            $table->float('Call_Duration')->nullable();
            $table->integer('Telesrvc')->nullable();
            $table->integer('LOCATION')->nullable();
            $table->integer('B_Num')->index()->nullable();
            $table->integer('cell_id')->nullable();
            $table->integer('IMEI')->nullable();
            $table->integer('TAC')->nullable();
            $table->string('DESTINATION_CAT')->nullable();
            $table->string('MU_HANDSET_DUAL_SIM')->nullable();
            $table->string('MU_Device_type_Segment')->nullable();
            $table->string('MU_HANDSET_MOBILE_TECH')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('c_d_r_s');
    }
};
