// Command ai-swap is a model-swapping proxy and chat UI for local inference
// engines (Sushi, ds4, Strata, llama.cpp, ...) across one or more machines.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/hofftodd/ai-swap/internal/cluster"
	"github.com/hofftodd/ai-swap/internal/config"
	"github.com/hofftodd/ai-swap/internal/scheduler"
	"github.com/hofftodd/ai-swap/internal/server"
	"github.com/hofftodd/ai-swap/web"
)

func main() {
	cfgPath := flag.String("config", "config.yaml", "path to config file")
	listen := flag.String("listen", "", "override node.listen")
	check := flag.Bool("check", false, "validate the config, print launch commands, and exit")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := run(*cfgPath, *listen, *check, log); err != nil {
		fmt.Fprintln(os.Stderr, "ai-swap:", err)
		os.Exit(1)
	}
}

func run(cfgPath, listen string, check bool, log *slog.Logger) error {
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return err
	}
	if listen != "" {
		cfg.Node.Listen = listen
	}
	mgr, err := scheduler.New(cfg, log)
	if err != nil {
		return err
	}
	if check {
		return printPlan(cfg, mgr)
	}
	cl, err := cluster.New(cfg.Peers, log)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go mgr.RunReaper(ctx, 15*time.Second)
	go cl.Run(ctx, 10*time.Second)

	srv := &http.Server{
		Addr:              cfg.Node.Listen,
		Handler:           server.New(cfg, mgr, cl, web.FS(), log).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	errc := make(chan error, 1)
	go func() {
		log.Info("ai-swap listening", "addr", cfg.Node.Listen, "node", cfg.Node.Name,
			"models", len(cfg.Models), "peers", len(cfg.Peers))
		errc <- srv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		if !errors.Is(err, http.ErrServerClosed) {
			mgr.StopAll()
			return err
		}
	case <-ctx.Done():
	}
	log.Info("shutting down; stopping models")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	mgr.StopAll()
	return nil
}

func printPlan(cfg *config.Config, mgr *scheduler.Manager) error {
	fmt.Printf("node %q listen %s budget %.0f GB\n", cfg.Node.Name, cfg.Node.Listen, cfg.Node.MemoryBudgetGB)
	next := cfg.Node.PortRange[0]
	alloc := func() (int, error) { next++; return next - 1, nil }
	for _, name := range cfg.ModelNames() {
		inst, _ := mgr.Get(name)
		l, err := inst.Engine.Build(inst.Model, cfg.Engines[inst.Model.Engine], alloc)
		if err != nil {
			fmt.Printf("  %-28s ERROR %v\n", name, err)
			continue
		}
		note := ""
		if !inst.Engine.Supported() {
			note = " (not runnable on this OS; will route to a peer)"
		}
		fmt.Printf("  %-28s %-8s %5.0f GB  port %-5d %q%s\n",
			name, inst.Engine.Name(), inst.Model.MemoryGB, l.Port, l.Argv, note)
	}
	return nil
}
